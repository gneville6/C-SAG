#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:36:06 2019

@author: Carter
"""

#CV project test

import numpy as np
import cv2
from cv2 import SimpleBlobDetector
import glob 
import os
import scipy
import random
from collections import Counter
from PIL import Image, ImageOps


class ColorThresholdAttempt:
    def __init__(self):
        self.test = 4
        self.blue_threshold_values = (np.array([99,50,50]), np.array([115,255,255])) # lower values, upper values
        #self.green_threshold_values = (np.array([0,0,0]), np.array([90,255,255]))  # lower values, upper values
        self.green_threshold_values = (np.array([60,0,0]), np.array([90,255,255]))  # lower values, upper values
        self.white_threshold_values = (np.array([0,0,230]), np.array([255,255,255]))

        self.params = cv2.SimpleBlobDetector_Params()

        # Filter by Area.
        self.params.filterByArea = True
        self.params.filterByInertia = False
        self.params.filterByConvexity = False
        self.params.minArea = 0
        self.params.maxArea = 2000
        self.minDistBetweenBlobs = 1

    def preview(self, im):
        cv2.imshow("preview", im)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def save_image(self, image, name):
        cv2.imwrite(name,image)
    
    def threshold_color(self, im, threshold_values):

        # Convert BGR to HSV
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        # define range colors in HSV
        lower_threshold_value = threshold_values[0]
        upper_threshold_value = threshold_values[1]
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_threshold_value, upper_threshold_value)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(im,im, mask= mask)
        #self.preview(im)
        #self.preview(mask)
        return mask
    
    def open_close_image(self, binary_matrix, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        erosion = cv2.erode(binary_matrix,kernel,iterations = 2)
        dilation = cv2.dilate(erosion,kernel,iterations = 1)
        return dilation

    def resize_image(self, im, scale):
        """
        scales image
        """
        w = int(im.shape[0] * scale)
        h = int(im.shape[1] * scale)
        im = cv2.resize(im,(h,w))
        return im

    def process_images_from_folder(self, path):
        """
        given path, process all images in folder
        """
        lis = os.listdir(path)
        for item in lis:
            im = cv2.imread(path + '/' + item)
            im = self.resize_image(im, 0.25)
            im1, im2 = self.process_image(im)
            #self.preview(im2)

    def fill_in_glare(self, ROI):
        """ WIP currently unused function """
        green_mask = np.zeros(ROI.shape)
        green_mask[:,:,0] = 32
        green_mask[:,:,1] = 100
        green_mask[:,:,2] = 1
        green = np.asarray(green_mask, dtype = 'uint8')
        white = self.threshold_color(ROI, self.white_threshold_values)
        result = cv2.bitwise_and(green,ROI, mask= white)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if sum(result[i,j,:]) == 0:
                    result[i,j,:] = [255,255,255]
        result2 = cv2.bitwise_and(result,ROI)
        #blue_thresh = threshold_blue(ROI)
        #preview(blue_thresh) 
        return result2

    def get_boundary_points_from_contour(self, im, contours):
        area = []
        for i in range(len(contours)):
            area.append(cv2.contourArea(contours[i]))
            
        cont = sorted(contours, key = cv2.contourArea, reverse = True)
        cont_pic = im.copy()
        cv2.drawContours(cont_pic, cont[0], -1, (0,255,0), 3)
        # self.preview(cont_pic)
        
        bucket_cont = np.asarray(cont[0])

        #grab the max & min X-Y values of the largest green contour
        x_vals = []
        y_vals = []
        for i in range(len(bucket_cont)):
            x_vals.append(bucket_cont[i][0][0])
            y_vals.append(bucket_cont[i][0][1])            
        

        # self.RANSAC_slope_detection(cont_pic, x_vals, y_vals)

        # x_1 = max(x_vals)
        # x_max_ind = x_vals.index(max(x_vals))
        # y_1 = y_vals[x_max_ind]
        # cv2.circle(cont_pic,(y_1,x_1), 20, (0,0,255), -1)
        
        #crop additional space off of the max values
        #figure out how to transform image/homography to overhead view
        cut_pad = 40    
        x_max = max(x_vals) - cut_pad
        x_min = min(x_vals) + cut_pad
        y_max = max(y_vals) - cut_pad
        y_min = min(y_vals) + cut_pad

        return x_min, x_max, y_min, y_max

    def RANSAC_slope_detection(self, cont_pic, x_vals, y_vals):
        """ WIP!! compute slopes of boundary detection given image and contour points """
        # RANSAC to find outline of bucket
        slope = []
        for i in range(0,10000):
            rand = random.randint(0,len(x_vals)-1)
            x_1 = x_vals[rand]
            y_1 = y_vals[rand]
            rand = random.randint(0,len(x_vals)-1)
            x_2 = x_vals[rand]
            y_2 = y_vals[rand]
            if (x_2 - x_1) == 0:
                s = float('inf')
                cept = x_2
            else:
                s = (y_2 - y_1)/(x_2 - x_1)
                s = round(s)
                cept = y_1 - s*x_1
            slope.append((s,cept))

        c = Counter(slope)
        most = c.most_common(1)
        x1 = int(0)
        y1 = int(most[0][0][0]*x1 + most[0][0][1])
        x2 = int(cont_pic.shape[1]-1)
        y2 = int(most[0][0][0]*x2 + most[0][0][1])
        cv2.line(cont_pic,(x1,y1),(x2,y2),(0,0,255),5)

    def crop_image_to_box_region(self, im, mask=True):
        ''' input raw image, return region of image that has the box inside '''
        #step 1 gaussian blur
        if self.test == 0:
            im = cv2.GaussianBlur(im,(7,7), 7)
        
        #step 2 green threshold for box
        background_mask = self.threshold_color(im, self.green_threshold_values)
        #self.preview(background_mask)

        #find contours and select the contour with the greatest area
        im2, contours, hierarchy = cv2.findContours(background_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            x_min, x_max, y_min, y_max = self.get_boundary_points_from_contour(im2, contours)
            print(contours[0])
            #crop to the sized of the max values on that contour
            ROI = im[y_min:y_max,x_min:x_max]

            if mask:
                mask = np.zeros(im.shape).astype(im.dtype)
                cont = sorted(contours, key = cv2.contourArea, reverse = True)
                cv2.fillPoly(mask, [cont[0]], [255, 255, 255])
                mask_inverted = cv2.bitwise_not(mask)
                mask_inverted = mask_inverted[y_min:y_max,x_min:x_max]
                mask_inverted[mask_inverted[:,:, 0] > 128] = [0, 255, 0]
                mask = mask[y_min:y_max, x_min:x_max] + mask_inverted
                ROI = cv2.bitwise_and(ROI, mask)
                #self.preview(ROI)
        else:
            ROI = im
        return ROI

    def remove_small_blobs(self, image_mask, image):
        verbose = 0

        size = image.shape
        desired_size = 1000
        delta_w = desired_size - size[1]
        delta_h = desired_size - size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)
        image_mask = cv2.copyMakeBorder(image_mask, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=color)

        proceesed_image = cv2.bitwise_and(image, image, mask=image_mask)
        #return proceesed_image, image_mask

        image_mask = cv2.cvtColor(cv2.bitwise_not(image_mask), cv2.COLOR_GRAY2RGB)

        first = 1
        round = 0
        while first or ( len(keypoints) != 0 and np.max(keypoint_size) > 10 and round < 1000 ):
            round += 1
            first = 0
            detector = cv2.SimpleBlobDetector_create(self.params)
            keypoints = detector.detect(image_mask)
            keypoint_size = []
            for x in range(len(keypoints)):
                keypoint_size.append(keypoints[x].size)
                img = cv2.circle(image_mask, (np.int(keypoints[x].pt[0]), np.int(keypoints[x].pt[1])),
                             radius=np.int(keypoints[x].size/2), color=(255,255,255), thickness=-1)
        if verbose == 1:
            im_with_keypoints = cv2.drawKeypoints(image_mask, keypoints, np.array([]), (0, 0, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            self.preview(im_with_keypoints)
            im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            self.preview(im_with_keypoints)


        image_mask = cv2.bitwise_not(cv2.cvtColor(image_mask, cv2.COLOR_RGB2GRAY))
        kernelD = np.ones((5, 5), np.uint8)
        kernelE = np.ones((1, 1), np.uint8)
        #image_mask = cv2.erode(image_mask, kernelE, iterations=1)
        image_mask = cv2.dilate(image_mask, kernelD, iterations=8)


        proceesed_image = cv2.bitwise_and(image, image, mask=image_mask)
        #self.preview(proceesed_image)
        return proceesed_image, image_mask

    def threshold_screw_images(self, ROI):
        """ Given image (assume its region of interest with only box region),

        return image of background blacked out to only contain screws and no color mask version"""

        green_thresh = self.threshold_color(ROI, self.green_threshold_values)
        print(green_thresh.shape)
        #self.preview(green_thresh)
        green_thresh_inverted = cv2.bitwise_not(green_thresh)
        if self.test == 0:
            green_thresh_open_close = self.open_close_image(green_thresh_inverted, 4)
        elif self.test == 1:
            green_thresh_open_close = self.open_close_image(green_thresh_inverted, 1)
        elif self.test == 2:
            green_thresh_open_close = self.open_close_image(green_thresh_inverted, 2)
        elif self.test == 3:
            green_thresh_open_close = self.open_close_image(green_thresh_inverted, 3)
        elif self.test == 4:
            green_thresh_open_close = green_thresh_inverted
        #green_thresh_inverted = morphOps(green_thresh_inverted, 5)
        #self.preview(green_thresh_open_close)

        res, green_thresh_open_close = self.remove_small_blobs(green_thresh_open_close, ROI)
        #self.preview(res)
        return res, green_thresh_open_close

    def process_image(self, im):
        """ given opencv/numpy image, run the processing method 
        
        return two images - only screws with color and only screws masked
         """
        ROI = self.crop_image_to_box_region(im)
        #self.preview(ROI)
        only_screws_im, masked_only_screws_im = self.threshold_screw_images(ROI)
        #self.preview(only_screws_im)

        return only_screws_im, masked_only_screws_im

    def process_image_just_crop(self, im, mask=False):
        """ given opencv/numpy image, run the processing method

        return two images - only screws with color and only screws masked
         """
        ROI = self.crop_image_to_box_region(im)

        if ROI is None:
            ROI  = np.zeros([800,800,3])

        size = ROI.shape
        desired_size = 800
        delta_w = desired_size - size[1]
        delta_h = desired_size - size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        ROI = cv2.copyMakeBorder(ROI, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)
        print(ROI.shape)

        # self.preview(ROI)
        #only_screws_im, masked_only_screws_im = self.threshold_screw_images(ROI)
        # self.preview(only_screws_im)

        return ROI

    def process_image_just_crop(self, im, threshhold, mask=False):
        """ given opencv/numpy image, run the processing method

        return two images - only screws with color and only screws masked
         """
        ROI = self.threshold_color(im, threshold_values=threshold)

        if ROI is None:
            ROI  = np.zeros([800,800,3])

        size = ROI.shape
        desired_size = 800
        delta_w = desired_size - size[1]
        delta_h = desired_size - size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        ROI = cv2.copyMakeBorder(ROI, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)
        print(ROI.shape)

        # self.preview(ROI)
        #only_screws_im, masked_only_screws_im = self.threshold_screw_images(ROI)
        # self.preview(only_screws_im)

        return ROI

    def process_image_crop_threshold(self, im):
        """ given opencv/numpy image, run the processing method

        return two images - only screws with color and only screws masked
         """
        ROI = self.crop_image_to_box_region(im)
        ROI = self.threshold_screw_images(ROI)
        # self.preview(ROI)
        #only_screws_im, masked_only_screws_im = self.threshold_screw_images(ROI)
        # self.preview(only_screws_im)

        return ROI


    def count_number_of_screws(self, im):
        """ Given unprocessed image, count the number of screws"""
        only_screws_im, masked_only_screws_im = self.process_image(im)
        im2, contours, hierarchy = cv2.findContours(masked_only_screws_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)
        print(num_contours)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(only_screws_im,'Number of screws:' + str(num_contours),(50,50), font, 1,(255,255,255),2,cv2.LINE_AA)
        
        #self.preview(only_screws_im)

        return num_contours

if __name__ == '__main__':
    attempt = ColorThresholdAttempt()
    # attempt.run()
    path = 'processed/Test_set'
    # attempt.process_images_from_folder(path)
    for j in os.scandir(path):
        path1 = j.path
        print(path1)
        for f in os.scandir(path1):
            if not f.is_file():
                #wtf
                continue
            im = cv2.imread(f.path)
            print(f.path)
            im = attempt.resize_image(im, 1.00)
            im1 = attempt.process_image_just_crop(im)
            im1 = attempt.process_image_light(im)
            #attempt.save_image(im1, f.path)
            #attempt.preview(im1)
    # attempt.count_number_of_screws(im)
