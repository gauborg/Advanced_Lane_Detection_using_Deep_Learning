import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import natsort
import pickle


# Thresholding functions
# since we have evaludated earlier that HLS gives good image filtering results

# this function returns a bindary image
# this functions accepts a grayscale image as input
def pixel_intensity(img, thresh = (0, 255)):

    # THIS FUNCTION WORKS ONLY ON GRAYSCALE IAMGES!!!
    # 1. apply threshold
    intensity_image = np.zeros_like(img)
    # scaled_pixel = np.uint8(255*img/255)
    # 2. create a binary image
    intensity_image[(img >= thresh[0]) & (img <= thresh[1])] = 1
    return intensity_image


# this function accepts a HLS format image
def lightness_select(img, thresh = (120,255)):
    
    # 2. Apply threshold to lightness channel
    l_channel = img[:,:,1]
    # 3. Create empty array to store the binary output and apply threshold
    lightness_image = np.zeros_like(l_channel)
    lightness_image[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
    return lightness_image



# this function accepts a HLS format image
def saturation_select(img, thresh = (100,255)):

    # 2. apply threshold to saturation channel
    s_channel = img[:,:,2]
    # 3. create empty array to store the binary output and apply threshold
    sat_image = np.zeros_like(s_channel)
    sat_image[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return sat_image



# this function accepts a RGB format image
# function to create binary image sobel gradients in x and y direction
def abs_sobel_thresh(img, orient = 'x', sobel_kernel = 5, thresh = (0,255)):

    # 1. Applying the Sobel depending on x or y direction and getting the absolute value
    if (orient == 'x'):
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = sobel_kernel))
    if (orient == 'y'):
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = sobel_kernel))
    # 2. Scaling to 8-bit and converting to np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 3. Create mask of '1's where the sobel magnitude is > thresh_min and < thresh_max
    sobel_image = np.zeros_like(scaled_sobel)
    sobel_image[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sobel_image



# this function accepts a RGB format image
# function to check binary image of sobel magnitude
def mag_sobel(img, sobel_kernel=3, thresh = (0,255)):

    # 1. Applying the Sobel (taking the derivative)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 2. Magnitude of Sobel
    mag_sobel = np.sqrt(sobelx**2 + sobely**2)
    # 3. Scaling to 8-bit and converting to np.uint8
    scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
    # 4. Create mask of '1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    sobel_mag_image = np.zeros_like(scaled_sobel)
    sobel_mag_image[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sobel_mag_image



# this function accepts a RGB format image
# function to compute threshold direction
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # 1. Applying the Sobel (taking the derivative)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 2. Take absolute magnitude
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 3. Take Tangent value
    sobel_orient = np.arctan2(abs_sobely, abs_sobelx)
    # 4. Create mask of '1's where the orientation magnitude is > thresh_min and < thresh_max
    dir_image = np.zeros_like(sobel_orient)
    dir_image[(sobel_orient >= thresh[0]) & (sobel_orient <= thresh[1])] = 1
    return dir_image



# --------------------------------------------------------------------------------------------------------------------------------- #



### Combined Thresholding Function
def combined_threshold(img):

    # convert to hls format and extract channels
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls = cv2.GaussianBlur(hls,(5,5),cv2.BORDER_DEFAULT)
    s_channel = hls[:,:,2]

    # convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # apply Gaussian Blur with kernel size of 5
    gray_blurred = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)
    
    # -------------------------- CALL FUNCTIONS FOR THRESHOLDING HERE! ----------------------------- #

    # get pixel intensity binary image
    # IMPORTANT: THIS FUNCION ACCEPTS GRAYSCALE IMAGES ONLY!!!
    pixel_intensity_binary = pixel_intensity(gray, thresh = (150, 255))
    
    # applying thresholding and storing different filtered images
    l_binary = lightness_select(hls, thresh = (140, 255))
    s_binary = saturation_select(hls, thresh = (100, 255))

    ksize = 5
    gradx = abs_sobel_thresh(gray_blurred, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    gradx_s_channel = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    # assuming lanelines are always at an angle between 30 and 45 degree to horizontal
    dir_binary = dir_threshold(gray_blurred, sobel_kernel=ksize, thresh=(0.55, 0.7))


    # ---------------------------- FUNCTION CALLS FOR THRESHOLDING END ------------------------------ #


    # creating an empty binary image
    combined_binary = np.zeros_like(s_binary)
    combined_binary[((pixel_intensity_binary == 1) | (l_binary == 1))] = 1

    combined_binary1 = np.zeros_like(s_binary)
    combined_binary1[((pixel_intensity_binary == 1) & (l_binary == 1))] = 1

    # apply region of interest mask
    height, width = combined_binary.shape
    mask = np.zeros_like(combined_binary)
    
    # define the region as 
    region = np.array([[0, height-1], [840, 400], [1080, 400], [width-1, height-1]], dtype=np.int32)
    # print(region)
    cv2.fillPoly(mask, [region], 1)

    masked = cv2.bitwise_and(combined_binary, mask)
    
    # This section is only for saving the separated hls plots.
    # This is commented out after running it once...
    '''
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(25, 10))
    ax1.imshow(pixel_intensity_binary)
    ax1.set_title('Hue', fontsize=20)
    ax2.imshow(l_binary)
    ax2.set_title('Lightness', fontsize=20)
    ax3.imshow(s_binary)
    ax3.set_title('Saturation', fontsize=20)
    ax4.imshow(gradx)
    ax4.set_title('X-Gradient', fontsize=20)
    
    '''
    
    # save hls separation plots
    # plt.savefig(('./rought_work/trial_images/hls-plots-'+ i), cmap = 'gray') 
    
    # save individual images for HSL plots, gradients, magnitude and direction
    mpimg.imsave(('./trial_images/intensity-test8.jpg'), pixel_intensity_binary, cmap = 'gray')
    mpimg.imsave(('./trial_images/gray-test8.jpg'), gray, cmap = 'gray')
    mpimg.imsave(('./trial_images/gray-blurred-test8.jpg'), gray_blurred, cmap = 'gray')
    mpimg.imsave(('./trial_images/l_binary-test8.jpg'), l_binary, cmap = 'gray')
    mpimg.imsave(('./trial_images/s_binary-test8.jpg'), s_binary, cmap = 'gray')
    mpimg.imsave(('./trial_images/gradx-test8.jpg'), gradx, cmap = 'gray')
    mpimg.imsave(('./trial_images/gradx_s_channel-test8.jpg'), gradx_s_channel, cmap = 'gray')
    mpimg.imsave(('./trial_images/direction-test8.jpg'), dir_binary, cmap = 'gray')

    # saving combined thresholded binary image
    mpimg.imsave(('./trial_images/combined-l_OR_intensity.jpg'), combined_binary, cmap = 'gray')
    mpimg.imsave(('./trial_images/combined-l_AND_intensity.jpg'), combined_binary1, cmap = 'gray')
    

    # saving masked images
    mpimg.imsave(('./trial_images/masked-test8.jpg'), masked, cmap = 'gray')
    # end of saving images section, comment out above section after saving the images
    
    
    return masked




### ------------------------------------------- LOAD PICKLE -------------------------------------------------- ###

file = open('../pickle/dist_pickle.p', 'rb')
# dump information to that file
data = pickle.load(file)
# close the file
file.close()
mtx, dst = data.values()


# ------------------------------------------------------------------------------------------------------------------- #


image = mpimg.imread("../test_images/test8.jpg")
# undistort the image
undist = cv2.undistort(image, mtx, dst, None, mtx)
masked_output = combined_threshold(image)

