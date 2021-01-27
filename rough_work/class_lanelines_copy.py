import numpy as np
import os
import glob
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import random


# class to store the characteristics of every laneline
class LaneLines():

    # constructor
    def __init__(self, binary_warped, prev_avg_left_fit, prev_avg_right_fit, prev_left_fit, prev_right_fit):
        # incoming binary image
        self.binary_warped = binary_warped
        # creating the array for binary
        self.ploty = np.linspace(0, self.binary_warped.shape[0]-1, self.binary_warped.shape[0])
        # no of windows
        # previous left and right fits which worked
        self.left_fit = prev_left_fit
        self.right_fit = prev_right_fit
        # average of past 10 fits
        self.avg_left_fit = prev_avg_left_fit
        self.avg_right_fit = prev_avg_right_fit


    # function for detecting lanelines manually
    def find_lane_pixels(self):

        # This part creates the histogram if lanelines are not detected in the previous iteration
        # Take a histogram of the bottom half of the image
        histogram = np.sum(self.binary_warped[self.binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((self.binary_warped, self.binary_warped, self.binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 15
        # Set the width of the windows +/- margin
        margin = 120
        # Set minimum number of pixels found to recenter window
        minpix = 20

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(self.binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = self.binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):

            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.binary_warped.shape[0] - (window+1)*window_height
            win_y_high = self.binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # VISUALIZATION
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
                    
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
                    
            # If we find > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        ##############################################################################################
        ############## THIS SECTION REMOVE OUTLIER PIXELS USING RMV_OUTLIERS FUNCTION ################
        # remove outlier points based on x coordinates
        leftx, outlier_list_left = self.rmv_outliers(leftx, 2)
        # remove the same points identified as outliers in lefty array
        lefty = np.delete(lefty, outlier_list_left)
        # remove outlier points based on x coordinates
        rightx, outlier_list_right = self.rmv_outliers(rightx, 2)
        # remove the same points identified as outliers in rightx array
        righty = np.delete(righty, outlier_list_right)
        
        ##############################################################################################
        ##############################################################################################
        ################################### FOR BAD LANELINES ########################################

        # if bad lanelines are detected
        if (leftx.size == 0 and rightx.size == 0):
            # bad lines are detected, lets commpute weighted average of previous estimates
            self.left_fit = self.avg_left_fit
            self.right_fit = self.avg_right_fit
        
        # if the number of total pixels detected is less than 50
        # this makes the calculations less accurate and hence we use
        # 90 % of the previous left fit average and 10% of the current
        elif ((leftx.size == 0) or (lefty.size == 0)):

            '''
            if left laneline is not detected, we use average right fit of previous frames...
            since lanewidth is 3.7 meters, and 3.7 meters = 700 pixels in our image
            if laneline is not found, we offset the left lane equation by 700 pixels
            we use the current left fit for our laneline
            '''
            offset_l_fit = np.polyfit(righty, rightx, 2)

            '''
            since lanewidth is 3.7 meters, and 3.7 meters = 700 pixels in our image
            if laneline is not found, we offset the right lane equation by 700 pixels
            we use the current right fit for our laneline
            '''
            offset_l_fit[2] = offset_l_fit[2] + (self.avg_right_fit[2] - self.avg_left_fit[2])

            # gather previous estimates for left laneline
            prev_avg_left = self.avg_left_fit

            self.left_fit = np.add(0.5*offset_l_fit, 0.5*prev_avg_left)
            self.right_fit = np.polyfit(righty, rightx, 2)
                   
        elif ((rightx.size == 0) or (righty.size == 0)):
            
            # if right laneline is not detected, we use average right fit of previous frames...
            offset_r_fit = np.polyfit(lefty, leftx, 2)
            # since lanewidth is 3.7 meters, and 3.7 meters = 700 pixels in our image
            # if laneline is not found, we offset the left lane by 700 pixels
            # we use the current left fit for our laneline
            offset_r_fit[2] = offset_r_fit[2] + (self.avg_right_fit[2] - self.avg_left_fit[2])

            # gather previous estimates for right laneline
            prev_avg_right = self.avg_right_fit

            self.right_fit = np.add(0.5*offset_r_fit, 0.5*prev_avg_right)
            try:
                self.left_fit = np.add(lefty, leftx, 2)
            except TypeError:
                pass

        # when a good number of pixels are detected and lines can be fit
        else:

            # if lanelines pixels are found, we use current values of lefty, leftx, righty, rightx to calculate our latest laneline equations...
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)

            '''
            check if the right lane value is less than left intercept value
            if yes, then update the right intercept by adding 700 to the 
            left intercept value
            '''
            # this means that the detected line is really bad and we compute the average of past fits
            if ((self.right_fit[2] < self.left_fit[2]) or (abs(self.right_fit[2]-self.left_fit[2]) > 750 )):
                
                # calculate offset intercept
                right = np.add(0.5*self.avg_left_fit, 0.5*self.left_fit)
                right[2] = 700 + self.avg_left_fit[2]

        # Generate x and y values for plotting
        try:
            self.left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
            self.right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]
        except IndexError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            print('Reverting to average of previous estimates')
            pass

        ## Visualization ##
        ## Uncommment only when running on test images"
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        
        return out_img, self.left_fit, self.right_fit


    def measure_curvature(self):
        
        '''
        Calculates the curvature of polynomial functions in pixels.
        Define y-value where we want radius of curvature
        We'll choose the maximum y-value, corresponding to the bottom of the image
        '''
        y_eval = np.max(self.ploty)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        left_fit_cr = np.polyfit(self.ploty*ym_per_pix, self.left_fitx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.ploty*ym_per_pix, self.right_fitx*xm_per_pix, 2)
        
        ##### Implement the calculation of R_curve in pixels (radius of curvature) #####
        left_curverad = (1 + ((2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])  ## Implement the calculation of the left line here
        right_curverad = (1 + ((2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        # Calculate the center offset of the vehicle
        lane_center = (self.right_fitx[self.binary_warped.shape[0]-1]-self.left_fitx[self.binary_warped.shape[0]-1])/2
        offset_in_pixels = abs(lane_center - (self.binary_warped.shape[0]/2))
        offset = offset_in_pixels * xm_per_pix

        return offset, left_curverad, right_curverad


    # function to remove outliers from an array
    def rmv_outliers(self, input_array, no_of_std_deviations=2):

        # calculate average of all elements
        mean = np.mean(input_array)
        # get the standard deviation
        std_deviation = np.std(input_array)

        dist_from_mean = abs(input_array - mean)
        
        outlier_indices = np.argwhere(dist_from_mean > no_of_std_deviations*std_deviation)
        length = outlier_indices.shape[0]

        # this will give us the indices of pixels which are outliers
        outlier_indices = outlier_indices.reshape((length,))
        
        # this gives the inlier pixels array
        out_array = np.delete(input_array, outlier_indices)

        return out_array, outlier_indices

