# code for extracting images from a video stream

'''
Description:

This code extracts images from given video files. Keep the video files in folder
"../data_for_lane_detection/new_videos/" one level up from the current directory.

The code will create a separate folder for saving images in the folder
"data for lane detection". This will save images from all the video files
starting with the id 1,000,001 and so on.

Author: Gaurav Borgaonkar
Date: 15 June 2020
Update Date: 03 August, 2020
Update: Updated info and prints the statusbar of extraction

'''

import os
import cv2
import glob
import moviepy.editor as mpy
import time
import progressbar
import pickle

'''
### ------------------------- LOAD PICKLE HERE ------------------------------ ###

pickle_file = open('pickle/dist_pickle.p', 'rb')
# dump information to that file
data = pickle.load(pickle_ile)
# close the file
pickle_file.close()
mtx, dst = data.values()

### ----------------------------- PICKLE READ ------------------------------- ###
'''

file_list = glob.glob("../data_for_lane_detection/new_videos/*")
total_videos = len(file_list)

# image index number for a million range
i = 1000001

# parameter to count the number of frames which could not be extracted
error_frame = 0

print(f"\nImages will be extracted from {total_videos} video files...\n")
print("Image extraction started...\n")
time.sleep(1.0)

# create a directory with the same name and '-images' for saving images
try:
    os.mkdir("../data_for_lane_detection/images")
    print("Created a new folder named images ...")
except FileExistsError:
    print("Folder with name images already exists!!!")
    print("Warning!!! - Extracted images will be merged with the contents of the existing folder!\n")
    pass

time.sleep(1.0)
for item in file_list:

    print("Working on file " + item)

    # Opens the Video file
    # code for extracting frames starts here ...
    video = cv2.VideoCapture(item)
    video_clip = mpy.VideoFileClip(item)
    frames = int(video_clip.fps * video_clip.duration)

    # code to print status
    frames_per_status_update = int(frames/50)  # update status for evey 2%

    bar = progressbar.ProgressBar(maxval=50, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    # no of processed frames
    processed_frames = 0
    # one set of frames processed
    processed_sets = 0
    # number of frames that could not be processed
    error_frame = 0
    
    while(video.isOpened()):
        # extract frame from the video file
        ret, frame = video.read()
        if(ret == True):
            # change resolution by adjusting fx and fy values
            frame = cv2.resize(frame, None, fx = 1.0, fy = 1.0, interpolation = cv2.INTER_CUBIC)
            # path to write images to ...
            cv2.imwrite('../data_for_lane_detection/images/' + 'image' + str(i) + '.jpg', frame)
            # increase image sequence number
            i += 1
        else:
            error_frame += 1
            break
        
        # code for printing extraction statuses
        # increment current frame
        processed_frames = processed_frames + 1

        if(processed_frames == frames_per_status_update):
            processed_sets += 1
            #print(processed_sets)
            try:
                bar.update(processed_sets)
            except ValueError:
                pass
            processed_frames = 0
        

    bar.finish()
    print(f"{error_frame-1} frames could not be processed!")

    # release the current video file
    video.release()
    cv2.destroyAllWindows()

    print(f"Extraction complete for file {item} ...")
    print(f"{frames} extracted from file {item} ...\n")

    # end of for loop


# print the summary
images = glob.glob("../data_for_lane_detection/images/image*")
num_images = (len(images))

print(f"{num_images} images extracted from {total_videos} video files ...")
print("Done!")



