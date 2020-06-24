# code for extracting images from a video stream

'''
Description: This code extracts images from a given video stream.
Author: Gaurav Borgaonkar
Date: 15 June 2020

'''

import os
import cv2
 
# Opens the Video file
filename = "../data_for_lane_detection/videos/video1.mp4"

# create a directory with the same name and '-images' for saving images
try:
    os.mkdir("images")
    print("Created a new folder named images ...")
except FileExistsError:
    print("Folder with name images already exists!!!")
    print("Images will be merged with the contents of the existing folder!")
    pass

# code for extracting frames starts here ...
video = cv2.VideoCapture(filename)

# image number sequence to start with
i = 1000001

while(video.isOpened()):
    
    ret, frame = video.read()
    if(ret == True):
        frame = cv2.resize(frame, None, fx = 1.0, fy = 1.0, interpolation = cv2.INTER_CUBIC)
        # path to write images to ...
        cv2.imwrite('../data_for_lane_detection/images/' + 'image' + str(i) + '.jpg', frame)
        i += 1
    else:
        break


video.release()

cv2.destroyAllWindows()

