# Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./examples/example_output.jpg)

### Requirements for running this project -

1. OpenCV libraries
2. NumPy libraries
3. Natsort libraries
4. Glob libraries

---

## Problem Statement

Using computer vision principles and OpenCV library functions, develop a robust algorithm to identify and draw lanelines in a given video stream.

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called *`camera_cal`*.  The images in *`test_images`* are for testing the pipeline on single frames.

---
## Image Pipeline

### Computing the Camera Calibration Matrix

**1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.**

The code for this section is in the ***pipeline_images.ipynb*** file textbox 2. Here, we execute the camera calibration matrix by providing the number of nx and ny, i.e. the number of chessboard corners in our images.

Camera images have a natural distortion present in them because of lense curvature. This curvature makes the images look distorted (fish-eye effect). To correct for this distortion, we perform camera calibration. We start by preparing two lists *imgpoints* and *objpoints* for storing the x and y co-ordinates of the detected corners in our chessboard images respectively. So, we will map the co-ordinates of the distorted image corners, i.e. *imgpoints* with real world undistorted corners, i.e. *objpoints*.

We then use the We do this by using the following functions from the OpenCV library -

[**cv2.findChessboardCorners()**](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#cv2.findChessboardCorners) for detecting chessboard corners

[**cv2.drawChessboardCorners()**](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#cv2.drawChessboardCorners) for drawing the detected chessboard corners

We use [**cv2.calibrateCamera()**](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#calibratecamera) to get the distortion coefficients *dist* and the camera calibration matrix *mtx* values. We use these values in function [**cv2.undistort()**](https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#undistort) to get undistorted image.

Here is an example of undistorted image -

![image](readme_images/test_undistortion.jpg)

We also get the perspective transform output for a chessboard image. Here is the output for drawn corners, undistorted and transformed image.

![image](readme_images/perspective_transform_output.jpg)

---

## Pipeline (for laneline test images)

**2. Provide an example of a distortion-corrected image.**

We again apply above listed principles to our test images and get the undistorted images saved. Here is an example -

![image](readme_images/undistorted_straight_lines1.jpg)

Other images can be found in the folder *output_images/test_images_undistorted*.

**3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.**

Text boxes 5 and 6 include the basic thresholding and combined function definitions respectively.

I have written separate functions for applying thresholds based on hue, lightness, separation channels and sobel gradients, magnitude and direction. In addition to these functions, I have also included a combined thresholding function which combines different thresholded images together to more robustly visualize lanelines. I implemented the color transform from RGB to HLS in the combined thresholding function and submitted the HLS formatted image as input to the individual thresholding functions.
I combined thresholds from x-direction gradient, lightness and saturation thresholded binary images to get a combined thresholding image as shown below:

```python
combined_binary[((l_binary == 1) & (s_binary == 1)) & ((gradx == 1) | (s_binary == 1))] = 1
```
I also applied region of interest mask to isolate only the bottom region of the image where the lanelines are always located. Here is the code for the same -

```python
# apply region of interest mask
height, width = combined_binary.shape
mask = np.zeros_like(combined_binary)
region = np.array([[0, height-1], [int(width/2), int(height/2)], [width-1, height-1]], dtype=np.int32)
# print(region)
cv2.fillPoly(mask, [region], 1)
```

Here is an example of a thresholded and masked image for test image *straight_lines1.jpg* -

![image](readme_images/masked-test1.jpg)

More images are saved in the *output_images/test_images_masked* folder.


**4. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.**

We define a function *perspective_view* in textbox 7 which takes as input our thresholded binary image. This function applies a perspective transform on the image. This gives us a bird's eye view of road so that the lanelines appear from the top and parallel. We select the endpoints of the laneline from a image approximately as the *src* points. We specify some destination *dst* points in the warped image so that our laneline will appear as parallel.



After this, we use the function [cv2.perspectiveTransform()](https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#getperspectivetransform) from OpenCV library by providing *src* and *dst* points as the inputs. This function calculates the 3*3 transformation matrix. We use this transformation matrix in a function called [cv2.warpPerspective()](https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#warpperspective) to get the warped (transformed iamge) of our lanelines.

Here is the code I used in the *perspective_view()* function -

```python
# image points extracted from image approximately
bottom_left = [210, 720]
bottom_right = [1100, 720]
top_left = [570, 470]
top_right = [720, 470]
src = np.float32([bottom_left, bottom_right, top_right, top_left])

pts = np.array([bottom_left, bottom_right, top_right, top_left], np.int32)
pts = pts.reshape((-1, 1, 2))
# create a copy of original img
imgpts = img.copy()
cv2.polylines(imgpts, [pts], True, (255, 0, 0), thickness=3)

# choose four points in warped image so that the lines should appear as parallel
bottom_left_dst = [320, 720]
bottom_right_dst = [920, 720]
top_left_dst = [320, 1]
top_right_dst = [920, 1]

dst = np.float32([bottom_left_dst, bottom_right_dst, top_right_dst, top_left_dst])
# apply perspective transform
M = cv2.getPerspectiveTransform(src, dst)
# compute inverse perspective transform
Minv = cv2.getPerspectiveTransform(dst, src)
# warp the image using perspective transform M
warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
```
Here is an example of a perspective transform image along with the original image. It shows us the original image along with undistorted, original warped and binary warped images.

![image](readme_images/perspective_transform_test2.jpg)

Additional images of perspective transform can be found in the folder *output_images/test_images_binary_warped*. (I have included only binary perspective transformed images here.)

**5. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?**

I have defined a class named *LaneLines* in a separate file *class_lanelines.py* which stores different parameters of detected such as x,y co-ordinates.

In this class, I have defined a function *find_lane_pixels()*. We use our binary thresholded perspective transformed image. We run sliding boxes approach to calculate sliding box approach to detect left and right laneline pixels in our image. We extract the x and y positions of the detected pixels and fit a 2nd order polynomial to the detected lanelines using the function [np.polyfit()](https://numpy.org/doc/1.18/reference/generated/numpy.polyfit.html?highlight=polyfit#numpy.polyfit).

```python
if ((leftx.size == 0) | (lefty.size == 0)):
    # if right laneline is not detected, we use average left fit of previous frames...
        temp_l_fit = self.avg_left_fit
        self.left_fit = temp_l_fit
        self.right_fit = np.polyfit(righty, rightx, 2)
        print('Reverting to average of previous estimates for left lane')
    elif ((rightx.size == 0) | (righty.size == 0)):
        # if right laneline is not detected, we use average right fit of previous frames...
        temp_r_fit = self.avg_right_fit
        self.right_fit = temp_r_fit
        self.left_fit = np.polyfit(lefty, leftx, 2)
        print('Reverting to average of previous estimates for right lane')
    else:
        # if lanelines pixels are found, we use current values of lefty, leftx,
        # righty, rightx to calculate our latest laneline equations...
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
```
Here is an example image of the laneline pixels detected using the sliding boxes approach.

![image](readme_images/test2-sliding-boxes.png)

**6. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.**

After fitting the 2nd order polynomials to both left and right lane pixels, we use the following formulae to determing the radius of curvature -

![image](readme_images/formulae.png)

I implemented a separate function *measure_curvature()* which calculates the left and right lane curvatures using the above formulae. This gives us the road curvature in pixels we convert it to road curvature in meters.

We calculate the lane center by subtracting the x fits for the left and right lanelines. In the next step, we calculate the center offset of the vehicle by subtracting the lane center from image center. This gives us the value of center offset in pixels. We use our pixels per meter value to convert this value from pixels to meters.

**7. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.**

![image](readme_images/test2-result.jpg)

---
## Video Pipeline

**Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!)**

The video pipeline is given in the ***video_lanelines.ipynb*** notebook. It uses almost the same pipeline as images. Only certain commands for plotting images have been commented out to run for video.

### [Link to the output video](videos_output)

---

## Discussion and Conclusion

**Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?**

### Problems/Issues faced during implementation:

I am listing a few issues I faced during the implementation of this project -

1. Changes in road texture and shadows from object, other vehicles is difficult to obtain with a static (fixed) thresholding. This made it difficult to detect lanelines in certain frames. To compensate for this shortcoming, I designed the code to use average lane fitting data of past 10-15 frames. Assuming a frame rate of 25 frames per second and hence elapsed time of 15/25 = 0.6 seconds, the actual lane curvature won't deviate too much from the calculated curvature.


### Potential Shortcomings

There are some shortcomings associated with this code.

1. This code and out image threshold is not robust enough to handle the effect of rain/snow, extreme lighting conditions, glare on the camera lens.
2. The current pipeline does not account for the effect of road bumps/potholes, banking, grade on our perspective transform.
3. This code can compensate for undetected lanelines in a couple of frames but is not robust in case of undetected lanelines for several frames continuously.
4. This code works perfectly for highway driving scenario where road curvature is larger but is not well suited for highly twisty roads.

### Improvements

1. To make better predictions in every frame (even in case of different road textures and shadows), a more robust thresholding with different filters (for hsv or rgb channels) can be used.
2. An advanced algorithm to compute the best fit from past several frames.
3. Use of estimation algorithms such as Kalman filters to improve our lane equations
