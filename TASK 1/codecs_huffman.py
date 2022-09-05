import cv2
import numpy as np 
import os 
import math
from glob import glob
from natsort import natsorted
from huffman import HuffmanCoding

def jpg_to_txt():
    """
    Function that creates a .txt file for each error frame, containing its pixel values
    """
    # filenames contains the path for every frame in the error_frames folder
    filenames = [frame for frame in glob("huffman_encoding/error_frames/*.jpg")]
    sorted = natsorted(filenames)

    # store the frames
    error_frames = []
    for error_frame in sorted:
        temp = cv2.imread(error_frame, cv2.IMREAD_GRAYSCALE)    # read the frames as grayscale
        error_frames.append(temp)
    # convert to numpy array
    error_frames = np.array(error_frames)

    # check if a folder to save the pixel values for each frame exists - if not, create it
    if not os.path.exists('huffman_encoding/error_frames/pixel_values)'):
        os.makedirs('huffman_encoding/error_frames/pixel_values')

    txt_names = []

    # for each error frame get its pixel values and store them in a file 
    for i in range(len(error_frames)):

        # save the pixel values in a text file
        one_d = error_frames[i].flatten()   # convert the 2d array into 1d
        
        # create the name of the txt file and add it in an array- it will be the same as the error frame's name
        temp = os.path.splitext(filenames[i])[0] + ".txt"
        temp2 = temp.split('\\')
        txt_name = temp2[1]
        txt_names.append(txt_name)

        a_file = open("huffman_encoding/error_frames/pixel_values/" + str(txt_name), "w")
        for value in one_d:
            a_file.write(str(value) + " ")  # write each pixel value in the txt file

        print("Created file: " + str(txt_name))

    return txt_names

def encode_decode(filename):
    path = "C:\\Users\\golem\\Desktop\\TASK 1\\huffman_encoding\\error_frames\\pixel_values\\" + filename
    H = HuffmanCoding(path)
    output_path = H.encode()
    H.decode(output_path)

    print("Finished with: " + filename)

if __name__ == "__main__":
    txt_names = jpg_to_txt()

    print(txt_names)

    for i in range(len(txt_names)):
        j = txt_names[i]
        encode_decode(j)
