# code for extracting images from a video stream

'''
Description: This code extracts images from a given video files ...
Author: Gaurav Borgaonkar
Date: 15 June 2020
Update Date: 23 June, 2020

'''

import os
import cv2
import glob
import tqdm

file_list = glob.glob("../data_for_lane_detection/videos/video*")
total_videos = len(file_list)

# image index number for a million range
i = 1000001

# parameter to count the number of frames which could not be extracted
error_frame = 0

print("Started the image extraction ...")
# create a directory with the same name and '-images' for saving images
try:
    os.mkdir("../data_for_lane_detection/images")
    print("Created a new folder named images ...")
except FileExistsError:
    print("Folder with name images already exists!!!")
    print("Images will be merged with the contents of the existing folder!")
    pass

for item in file_list:

    print("Working on file " + item)
    # Opens the Video file
    # code for extracting frames starts here ...
    video = cv2.VideoCapture(item)
    # time.sleep(0.01)
    while(video.isOpened()):
        # extract frame from the video file
        ret, frame = video.read()
        if(ret == True):
            frame = cv2.resize(frame, None, fx = 0.75, fy = 0.75, interpolation = cv2.INTER_CUBIC)
            # path to write images to ...
            cv2.imwrite('../data_for_lane_detection/images/' + 'image' + str(i) + '.jpg', frame)
            # increase image sequence number
            i += 1
        else:
            error_frame += 1
            break
    
    video.release()
    cv2.destroyAllWindows()

    print(f"Extraction complete for file {item} ...")
    print()
    # end of for loop


# print the summary
images = glob.glob("../data_for_lane_detection/images/image*")
num_images = (len(images))

print(f"{num_images} images extracted from {total_videos} video files ...")



