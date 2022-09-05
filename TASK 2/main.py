import cv2
import numpy as np 
import os
import math
from glob import glob
from natsort import natsorted

def cut_frames():
    """
    Function that cuts the video given into frames
    """
    frames_skipped = 10     # save 1 out of 10 frames
    cap = cv2.VideoCapture('car_moving.mp4')

    # check if a folder to save the frames exists - if not, create it
    try:
        if not os.path.exists('frames'):
            os.makedirs('frames')
    except OSError:
        print("Error creating folder!")
        
    currentFrame = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while (currentFrame < length):   # check if there are still frames in the video to read
        
        ret, frame = cap.read()     # capture frame-by-frame
                
        name = 'frames/frame' + str(math.trunc(currentFrame/10)) + '.jpg'

        # save only 1 out of 10 frames
        if (currentFrame // frames_skipped == currentFrame / frames_skipped):
            print('Creating...' + name)
            cv2.imwrite(name, frame)
        
        currentFrame += 1 
        
    cap.release()

if __name__ == "__main__":
    cut_frames()
