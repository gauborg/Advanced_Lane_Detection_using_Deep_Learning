import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
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
def saturation_select(img, thresh = (175,255)):

    # 2. apply threshold to saturation channel
    s_channel = img[:,:,2]
    mpimg.imsave(('./trial_images/s-channel-test8.jpg'), s_channel)
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
def mag_sobel(img, sobel_kernel=3, thresh = (30,100)):

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

# -------------------------------------------------------------------------------------------------- #



### Combined Thresholding Function
def combined_threshold(img):

    # convert to hls format and extract channels
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls = cv2.GaussianBlur(hls,(7,7),cv2.BORDER_DEFAULT)
    s_channel = hls[:,:,2]

    # convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # apply Gaussian Blur with kernel size of 5
    gray_blurred = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)
    
    # -------------------------- CALL FUNCTIONS FOR THRESHOLDING HERE! ----------------------------- #

    # get pixel intensity binary image
    # IMPORTANT: THIS FUNCION ACCEPTS GRAYSCALE IMAGES ONLY!!!
    pixel_intensity_binary = pixel_intensity(gray, thresh = (160, 255))
    
    # applying thresholding and storing different filtered images
    l_binary = lightness_select(hls, thresh = (140, 255))
    s_binary = saturation_select(hls, thresh = (100, 255))

    ksize = 7
    gradx = abs_sobel_thresh(gray_blurred, orient='x', sobel_kernel=ksize, thresh=(30, 150))
    # assuming lanelines are always at an angle between 30 and 45 degree to horizontal
    dir_binary = dir_threshold(gray_blurred, sobel_kernel=ksize, thresh=(0.55, 0.72))

    sobel_img = mag_sobel(gray, 3, (25, 100))


    # ---------------------------- FUNCTION CALLS FOR THRESHOLDING END ------------------------------ #


    # creating an empty binary image
    combined_binary_l_or_intensity = np.zeros_like(s_binary)
    combined_binary_l_or_intensity[((pixel_intensity_binary == 1) | (l_binary == 1))] = 1

    combined_binary_l_and_intensity = np.zeros_like(s_binary)
    combined_binary_l_and_intensity[((pixel_intensity_binary == 1) & (l_binary == 1))] = 1

    # combined binary from X gradient, Saturation and Lightness
    combined_binary2 = np.zeros_like(gradx)
    combined_binary2[((sobel_img == 1) & (gradx == 1))] = 1

    # apply region of interest mask
    height, width = combined_binary_l_or_intensity.shape
    mask = np.zeros_like(combined_binary_l_or_intensity)
    
    # define the region as 
    region = np.array([[0, height-1], [840, 400], [1080, 400], [width-1, height-1]], dtype=np.int32)
    # print(region)
    cv2.fillPoly(mask, [region], 1)

    masked = cv2.bitwise_and(combined_binary_l_or_intensity, mask)
    
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
    mpimg.imsave(('./trial_images/shiny_road_spl_case/intensity-shiny-road.jpg'), pixel_intensity_binary, cmap = 'gray')
    mpimg.imsave(('./trial_images/shiny_road_spl_case/gray-shiny-road.jpg'), gray, cmap = 'gray')
    mpimg.imsave(('./trial_images/shiny_road_spl_case/gray-blurred-shiny-road.jpg'), gray_blurred, cmap = 'gray')
    mpimg.imsave(('./trial_images/shiny_road_spl_case/l_binary-shiny-road.jpg'), l_binary, cmap = 'gray')
    mpimg.imsave(('./trial_images/shiny_road_spl_case/s_binary-shiny-road.jpg'), s_binary, cmap = 'gray')
    mpimg.imsave(('./trial_images/shiny_road_spl_case/gradx-shiny-road.jpg'), gradx, cmap = 'gray')
    
    mpimg.imsave(('./trial_images/shiny_road_spl_case/direction-shiny-road.jpg'), dir_binary, cmap = 'gray')
    mpimg.imsave(('./trial_images/shiny_road_spl_case/sobel_img-shiny-road.jpg'), sobel_img, cmap = 'gray')


    # saving combined thresholded binary image
    mpimg.imsave(('./trial_images/shiny_road_spl_case/combined-l_OR_intensity-shiny-road.jpg'), combined_binary_l_or_intensity, cmap = 'gray')
    mpimg.imsave(('./trial_images/shiny_road_spl_case/combined-l_AND_intensity-shiny-road.jpg'), combined_binary_l_and_intensity, cmap = 'gray')
    mpimg.imsave(('./trial_images/shiny_road_spl_case/combined-sobel_AND_gradx-shiny-road.jpg'), combined_binary2, cmap = 'gray')

    # end of saving images section, comment out above section after saving the images
    
    return masked

### ----------------------------------- END OF THRESHOLDING FUNCTIONS ---------------------------------------- ###





### -------------------------------------- PERSPECTIVE TRANSFORM --------------------------------------------- ###

# function for applying perspective view on the images
def perspective_view(img):

    img_size = (img.shape[1], img.shape[0])

    # image points extracted from image approximately
    bottom_left = [340, 825]
    bottom_right = [1480, 825]
    top_left = [895, 445]
    top_right = [1028, 445]

    src = np.float32([bottom_left, bottom_right, top_right, top_left])

    pts = np.array([bottom_left, bottom_right, top_right, top_left], np.int32)
    pts = pts.reshape((-1, 1, 2))
    # create a copy of original img
    imgpts = img.copy()
    cv2.polylines(imgpts, [pts], True, (255, 0, 0), thickness = 3)

    # choose four points in warped image so that the lines should appear as parallel
    bottom_left_dst = [600, 1080]
    bottom_right_dst = [1300, 1080]
    top_left_dst = [600, 1]
    top_right_dst = [1300, 1]

    dst = np.float32([bottom_left_dst, bottom_right_dst, top_right_dst, top_left_dst])

    # apply perspective transform
    M = cv2.getPerspectiveTransform(src, dst)

    # compute inverse perspective transform
    Minv = cv2.getPerspectiveTransform(dst, src)

    # warp the image using perspective transform M
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M, Minv


### ------------------------------------ END OF PERSPECTIVE TRANSFORM ---------------------------------------- ###



### ------------------------------------------- LOAD PICKLE -------------------------------------------------- ###

file = open('../pickle/dist_pickle.p', 'rb')
# dump information to that file
data = pickle.load(file)
# close the file
file.close()
mtx, dst = data.values()

### ---------------------------------------------------------------------------------------------------------- ###

image = mpimg.imread("./test8.jpg")

# save original image
mpimg.imsave(('./trial_images/original-test8.jpg'), image)

# save original
undist = cv2.undistort(image, mtx, dst, None, mtx)
mpimg.imsave(('./trial_images/undistorted-masked-test8.jpg'), undist)

# undistort the image
masked_output = combined_threshold(undist)

# saving masked images
mpimg.imsave(('./trial_images/masked-test8.jpg'), masked_output, cmap = 'gray')

binary_warped, M, Minv = perspective_view(masked_output)
mpimg.imsave(('./trial_images/binary_warped-test8.jpg'), binary_warped, cmap = 'gray')

### ---------------------------------------------------------------------------------------------------------- ###


### ------------------------------------ SHINY ROAD SURFACE SPECIAL CASE ------------------------------------- ###

image = mpimg.imread("./test-shiny-road.jpg")

# save original image
mpimg.imsave(('./trial_images/shiny_road_spl_case/original-test-shiny-road.jpg'), image)

# save original
undist = cv2.undistort(image, mtx, dst, None, mtx)
mpimg.imsave(('./trial_images/shiny_road_spl_case/undistorted-masked-shiny-road.jpg'), undist)

# undistort the image
masked_output = combined_threshold(undist)

# saving masked images
mpimg.imsave(('./trial_images/shiny_road_spl_case/masked-shiny-road.jpg'), masked_output, cmap = 'gray')

binary_warped, M, Minv = perspective_view(masked_output)
mpimg.imsave(('./trial_images/shiny_road_spl_case/binary_warped-shiny-road.jpg'), binary_warped, cmap = 'gray')

### ---------------------------------------------------------------------------------------------------------- ###
