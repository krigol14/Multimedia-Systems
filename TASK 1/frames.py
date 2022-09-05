import cv2
import numpy as np 
import os 
import math
from glob import glob
from natsort import natsorted

def cut_frames():
    """
    Function that cuts the video given into frames.
    """
    frames_skipped = 10     # save 1 out of 10 frames
    cap = cv2.VideoCapture('grayscale.mp4')

    # check if a folder to save the frames exists - if not, create it
    try:
        if not os.path.exists('huffman_encoding/frames'):
            os.makedirs('huffman_encoding/frames')
    except OSError:
        print("Error creating folder!")
        
    currentFrame = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while (currentFrame < length):   # check if there are still frames in the video to read
        
        ret, frame = cap.read()     # capture frame-by-frame
                
        name = 'huffman_encoding/frames/frame' + str(math.trunc(currentFrame/10)) + '.jpg'

        # save only 1 out of 10 frames
        if (currentFrame // frames_skipped == currentFrame / frames_skipped):
            print('Creating...' + name)
            cv2.imwrite(name, frame)
        
        currentFrame += 1 
        
    cap.release()

def predict_frames():
    """
    Function that predicts the next frames. In our case the prediction is very simple as we suppose 
    that the next frame is always the same as the previous one.
    """
    # filenames variable contains the path for every frame stored in the frames/ folder
    filenames = [img for img in glob("huffman_encoding/frames/*.jpg")]
    filenames = natsorted(filenames)

    # array to store the frames
    frames = []
    for frame in filenames:
        temp = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)    # read the frames as grayscale
        frames.append(temp)

    # convert to numpy array
    frames = np.array(frames)

    # check if a folder to store the predicted frames exists - if not, create it
    try:
        if not os.path.exists('huffman_encoding/predicted_frames'):
            os.makedirs('huffman_encoding/predicted_frames')
    except OSError:
        print("Error creating folder!")

    predicted_frames = []
    predicted_frames.insert(0, frames[0])   # the first frame is the same as the first actual frame

    # every predicted frame is the same as the previous actual frame
    for i in range(1, len(frames)):
        predicted_frames.insert(i, frames[i-1])

    # save the predicted frames in a seperate folder
    currentFrame = 0
    while (currentFrame < len(predicted_frames)):

        name = 'huffman_encoding/predicted_frames/p_frame' + str(currentFrame) + '.jpg'
        predicted = predicted_frames[currentFrame]

        print('Creating...' + name)
        cv2.imwrite(name, predicted)
        currentFrame += 1
    
    return frames, predicted_frames

def prediction_error(frames, predicted_frames):
    """
    Function to evaluate, display and save in a folder the frames that visualize the prediction error.
    """
    # check if a folder to store the error frames exists - if not, create it
    try:
        if not os.path.exists('huffman_encoding/error_frames'):
            os.makedirs('huffman_encoding/error_frames')
    except OSError:
        print("Error creating folder!")

    error_frames = []

    for i in range(len(frames)):

        # error = actual_frame(n) - predicted_frame(n)
        prediction_error = np.subtract(frames[i], predicted_frames[i])
        error_frames.insert(i, prediction_error)

    currentFrame = 0
    while (currentFrame < len(error_frames)):

        name = 'huffman_encoding/error_frames/error_frame' + str(currentFrame) + '.jpg'
        error = error_frames[currentFrame]

        print('Creating...' + name)
        cv2.imwrite(name, error)
        currentFrame += 1

def display(folder_name, title):
    filenames = [img for img in glob(folder_name + "/*.jpg")]
    sorted = natsorted(filenames)

    for frame in sorted:
        temp = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
        cv2.imshow(title, temp)
        cv2.waitKey(0)

if __name__ == "__main__":
    cut_frames()
    print("----------")
    frames, predicted_frames = predict_frames()
    print("----------")
    prediction_error(frames, predicted_frames)

    display("huffman_encoding/frames", "frames")
    display("huffman_encoding/predicted_frames", "predicted_frames")
    display("huffman_encoding/error_frames", "error_frames")
      