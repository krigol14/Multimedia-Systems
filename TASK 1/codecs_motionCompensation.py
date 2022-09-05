from genericpath import exists
import numpy as np
import cv2
import random
import time
import os
from natsort import natsorted
from glob import glob
from numpy.core.arrayprint import format_float_scientific
from motion_compensation import *

# filenames contains the path for every frame in the frames folder
filenames = [frame for frame in glob("huffman_encoding/frames/*.jpg")]
sorted = natsorted(filenames)
# store the frames
frames = []
for frame in sorted:
    temp = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)    # read the frames as grayscale
    frames.append(temp)
# convert to numpy array
frames = np.array(frames)

def encoder(iframe, target_frame):
    """
    Function that implements the encoding process using the predefined methods.
    """
    predicted_frame = create_predicted(iframe, target_frame)
    residual_frame = find_residual(target_frame, predicted_frame)

    return predicted_frame, residual_frame

def decoder(residual_frame, predicted_frame):
    """
    Function that implements the decoding process using the predefined methods.
    """
    reconstructed_frame = reconstruct_target(residual_frame, predicted_frame)

    return reconstructed_frame

def helper(previous_frame, next_frame):

    predicted_frame, residual_frame = encoder(previous_frame, next_frame)
    reconstructed_frame = decoder(residual_frame, predicted_frame) 
    
    return predicted_frame, residual_frame, reconstructed_frame

def main():
    """
    Main function.
    """
    # check if the folders to store the frames exist - if not, create them
    exists = os.path.isdir("motion_compensation")
    if not exists:
        os.mkdir("motion_compensation")

    exists = os.path.isdir("motion_compensation/pframes")
    if not exists:
        os.mkdir("motion_compensation/pframes")
    
    exists = os.path.isdir("motion_compensation/target_frames")
    if not exists:
        os.mkdir("motion_compensation/target_frames")
        
    exists = os.path.isdir("motion_compensation/residual_frames")
    if not exists:
        os.mkdir("motion_compensation/residual_frames")

    exists = os.path.isdir("motion_compensation/reconstructed_frames")
    if not exists:
        os.mkdir("motion_compensation/reconstructed_frames")   

    predicted_frame, residual_frame, reconstructed_frame = helper(frames[0], frames[1])

    for i in range(len(frames) + 1):
        previous_frame = predicted_frame
        next_frame = frames[i+1]
        predicted_frame, residual_frame, reconstructed_frame = helper(predicted_frame, frames[i+1])

        # save the images in a seperate folder
        cv2.imwrite("motion_compensation/pframes/pframe" + str(i) + ".jpg", previous_frame)
        cv2.imwrite("motion_compensation/target_frames/target_frame" + str(i) + ".jpg", next_frame)
        cv2.imwrite("motion_compensation/residual_frames/residual_frame" + str(i) + ".jpg", residual_frame)
        cv2.imwrite("motion_compensation/reconstructed_frames/reconstructed_frame" + str(i) + ".jpg", reconstructed_frame)

if __name__ == "__main__":
    main()
