# code for extracting images from a video stream

'''
Description: This code extracts images from a given video stream.
Author: Gaurav Borgaonkar
Date: 15 June 2020

'''

import os
import cv2
 
# Opens the Video file
filepath = "./monument-valley-1.mp4"

# split the filename
filename_w_ext = os.path.basename(filepath)
filename, file_extension = os.path.splitext(filename_w_ext)

# assign a folder name
folder_name = filename + '-images'

# create a directory with the same name and '-images' for saving images
try:
    os.mkdir(folder_name)
    print(f"Created a new folder named {folder_name} ...")
except FileExistsError:
    print(f"Folder with name {filename} already exists!!!")
    print("Images will be merged with the contents of the existing folder!")
    pass

current_dir = os.getcwd()

# append path of folder for saving images
images_path = os.path.join(current_dir, folder_name)
print(images_path)


# code for extracting frames starts here ...
video = cv2.VideoCapture(filename)

# image number sequence to start with
i = 1000001

error_frames = 0

while(video.isOpened()):
    ret, frame = video.read()
	
	try:
        frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)
		# path to write images to ...
		cv2.imwrite(images_path + './image' + str(i) + '.jpg', frame)
		i += 1
		
    except cv2.error:
        pass
		error_frames += 1
		
 
video.release()
cv2.destroyAllWindows()


