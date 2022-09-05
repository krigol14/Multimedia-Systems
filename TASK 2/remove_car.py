from re import search
from typing import final
import numpy as np
from natsort import natsorted
from glob import glob
import cv2
import os
import math

# filenames contains the path for every frame in the frames folder
filenames = [frame for frame in glob("frames/*.jpg")]
sorted = natsorted(filenames)
# store the frames
frames = []
for frame in sorted:
    temp = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)    # read the frames as grayscale
    frames.append(temp)
# convert to numpy array
frames = np.array(frames)

# car_frames contains the path for every frame in the specific_area folder
car_frames = [frame for frame in glob("frames/specific_area/*.jpg")]
sorted = natsorted(car_frames)
# store the frames
car_frames = []
for frame in sorted:
    temp = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)    # read the frames as grayscale
    car_frames.append(temp)
# convert to numpy array
car_frames = np.array(car_frames)

# no_car contains the path for every frame in the no_car folder
no_car = [frame for frame in glob("frames/no_car/*.jpg")]
sorted = natsorted(no_car)
# store the frames
no_car = []
for frame in sorted:
    temp = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)    # read the frames as grayscale
    no_car.append(temp)
# convert to numpy array
no_car = np.array(no_car)

def get_car():
    """
    Function that cuts the region the car exists in the original video. 
    In this way it's simpler to remove the car from the video as the background seems to be the same in all frames.
    """
    try:
        if not os.path.exists('frames/specific_area'):
            os.makedirs('frames/specific_area')
    except OSError:
        print("Error creating folder!")

    for i in range(len(frames)):
        wanted_area = frames[i][272:448, 512:656]
        cv2.imwrite("frames/specific_area/car" + str(i) + ".jpg", wanted_area)

def divide_frame(frame):
    """
    Function that returns the number of vertical and horizontal macroblocks for the given frame.
    """
    height, width = frame.shape
    vertical_mblocks = int(height/ 16)
    horizontal_mblocks = int(width/ 16)
    total_mblocks = int(vertical_mblocks * horizontal_mblocks)

    return vertical_mblocks, horizontal_mblocks

def find_center(x, y):
    """
    Determines the center of a block whose top left corner coordinates are x, y.
    e.g. for the block whose top left corner coordinates are (0, 0), the center is (8, 8).
    """
    return(int(x + 8), int(y + 8))

def find_search_area(x, y, frame):
    """
    Returns the search area of a frame based to the coordinates given.
    - x, y are the top left corner's coordinates of the given macroblock.
    """
    height, width = frame.shape
    center_x, center_y = find_center(x, y)
    
    # ensure search area is within bounds
    # find the top left corner's coordinates of the search area
    search_x = max(0, center_x - 24)    
    search_y = max(0, center_y - 24)

    # slice frame within bounds to produce the search area
    search_area = frame[search_y:min(search_y + 48, height), search_x:min(search_x + 48, width)]

    return search_area

def get_block_zone(p, search_area, current_block):
    """
    Retrieves the block searched in the search area of the initial I-frame to be compared with the current_block of the current frame.
    - (x, y): coordinates of the current frame's macroblock center.
    """   
    px, py = p                          # coordinates of the macroblock's center
    px, py = px - 8, py - 8             # top left corner
    px, py = max(0, px), max(0, py)     # ensure within bounds

    a_block = search_area[py:py + 16, px:px + 16]

    # make sure the two blocks have the same shape
    try:
        assert a_block.shape == current_block.shape
    except:
        print("The blocks should have the same shape!")

    return a_block

def find_mad(current_block, a_block):
    """
    Returns Mean Absolute Difference between current_block and a_block.
    - current_block: current frame macroblock.
    - a_block: I-frame macroblock.
    """
    return np.sum(np.abs(np.subtract(current_block, a_block)))/ (current_block.shape[0] * current_block.shape[1])

def find_match(current_block, search_area):
    """
    Algorithm to find the best pair of blocks between the current frame's block and the I-frame's block. 
    Returns the macroblock from the I-frame's search area with the least Mean Absolute Difference.
    """
    step = 4
    sa_height, sa_width = search_area.shape
    # find the center of the I-frame's search area
    sa_centerY, sa_centerX = int(sa_height/2), int(sa_width/2)

    min_mad = float("+inf")
    minP = None

    while step >= 1:
        p1 = (sa_centerX, sa_centerY)
        p2 = (sa_centerX + step, sa_centerY)
        p3 = (sa_centerX, sa_centerY + step)
        p4 = (sa_centerX + step, sa_centerY + step)
        p5 = (sa_centerX - step, sa_centerY)
        p6 = (sa_centerX, sa_centerY - step)
        p7 = (sa_centerX - step, sa_centerY - step)
        p8 = (sa_centerX + step, sa_centerY - step)
        p9 = (sa_centerX - step, sa_centerY + step)

        # retrieve the nine search points
        pointList = [p1, p2, p3, p4, p5, p6, p7, p8, p9]

        for p in range(len(pointList)):
            # get the I-frame's macroblock
            a_block = get_block_zone(pointList[p], search_area, current_block)
            mad = find_mad(current_block, a_block)
            if mad < min_mad:
                min_mad = mad
                minP = pointList[p]
            
        step = int(step/2)

    px, py = minP                       # center of the macroblock with the least M.A.D.
    px, py = px - 8, py - 8             # top left corner
    px, py = max(0, px), max(0, py)     # ensure it's within bounds

    match = search_area[py:py + 16, px:px + 16]

    return match, px, py 

def find_motion(frame, target_frame):
    """
    Returns a matrix whose values represent the movement of each macroblock from frame_n to frame_n+1.
    The values of the matrix represent the motion vectors for the motion compensation algorithm we are using.
    """
    height, width = frame.shape
    vertical_mblocks, horizontal_mblocks = divide_frame(frame)

    pixels_moved = np.ones((height, width))
    bcount = 0

    for y in range(0, int(vertical_mblocks * 16), 16):
        for x in range(0, int(horizontal_mblocks * 16), 16):

            target_block = target_frame[y:y + 16, x:x + 16]
            search_area = find_search_area(x, y, frame)
            matching_block, px, py = find_match(target_block, search_area)

            target_position = (x, y)
            matching_position = (px, py)
            position_difference = math.sqrt((x-px)**2 + (y-py)**2)

            pixels_moved[y][x] = "{:.1f}".format(position_difference)

    return pixels_moved

def remove_car_helper(frame, target_frame):
    """
    Helper function to call it recursively from the remove_car() method and remove the car from the frames.
    """
    pixels_moved = find_motion(frame, target_frame)

    pixels_moved[pixels_moved < 100] = 0
    pixels_moved = pixels_moved.flatten()

    new_frame = frame

    for i in range(len(pixels_moved)):
        # it means that part of the block has moved, thus it's the car
        if pixels_moved[i] != 0:
            new_frame[i:i+16, i:i+16] = frame[i:i+16, i:i+16]
    
    return new_frame

def remove_car():
    """
    Function which removes the car from the frames that contain the specific area the car exists.
    """
    try:
        if not os.path.exists('frames/no_car'):
            os.makedirs('frames/no_car')
    except OSError:
        print("Error creating folder!")

    new_frame = remove_car_helper(car_frames[0], car_frames[1])

    for i in range(len(car_frames) + 1):
        removed_car = new_frame
        next_frame = car_frames[i + 1]
        new_frame = remove_car_helper(removed_car, frames[i + 1])

        cv2.imwrite("frames/no_car/no_car" + str(i) + ".jpg", new_frame)

def final_removement(no_car, frame):
    """
    Function that removes the car from the area it exists in the original frames of the video.
    """
    frame[272:448, 512:656] = no_car
    
    return frame

def frames_to_video():
    """
    Function that creates a video using the frames created that no longer contain the car.
    """
    fps = 1
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    filename = "no-car.mp4"
    video_writer = cv2.VideoWriter(filename, fourcc, fps, (1280, 720))
    for i in range(0, 18):
        if os.path.exists("frames/final/final" + str(i) + ".jpg"):
            img = cv2.imread(filename = "frames/final/final" + str(i) + ".jpg")
            cv2.waitKey(100)
            video_writer.write(img)
            print(str(i) + ".jpg done")
    video_writer.release()

if __name__ == "__main__":

    # get_car() 
       
    # remove_car()

    # try:
    #     if not os.path.exists('frames/final/'):
    #         os.makedirs('frames/final')
    # except OSError:
    #     print("Error creating folder!")

    # for i in range(len(frames)):
    #     final = final_removement(no_car[i], frames[i+1])
    #     cv2.imwrite("frames/final/final" + str(i) + ".jpg", final)

    frames_to_video()
