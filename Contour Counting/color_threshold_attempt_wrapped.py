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


def is_cv2():
    # if we are using OpenCV 2, then our cv2.__version__ will start
    # with '2.'
    return check_opencv_version("2.")
 
def is_cv3():
    # if we are using OpenCV 3.X, then our cv2.__version__ will start
    # with '3.'
    return check_opencv_version("3.")
 
def is_cv4():
    # if we are using OpenCV 3.X, then our cv2.__version__ will start
    # with '4.'
    return check_opencv_version("4.")
 
def check_opencv_version(major, lib=None):
    # if the supplied library is None, import OpenCV
    if lib is None:
        import cv2 as lib
        
    # return whether or not the current OpenCV version matches the
    # major version number
    return lib.__version__.startswith(major)


class ColorThresholdAttempt:
    def __init__(self):
        self.test = 4

        #HSV

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

        self.threshold_num = 500

    def preview(self, im):
        cv2.imshow("preview", im)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def save_image(self, image, name):
        cv2.imwrite(name,image)

    def threshold_red(self, img):
        img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # lower mask (0-10)
        lower_red = np.array([0,50,50])
        upper_red = np.array([10,255,255])
        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

        # upper mask (170-180)
        lower_red = np.array([170,50,50])
        upper_red = np.array([180,255,255])
        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

        # join my masks
        mask = mask0+mask1

        return mask
    
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

    def crop_image_to_box_region(self, im, mask=True):
        ''' input raw image, return region of image that has the box inside '''
        #step 1 gaussian blur
        if self.test == 0:
            im = cv2.GaussianBlur(im,(7,7), 7)
        
        #step 2 green threshold for box
        background_mask = self.threshold_color(im, self.green_threshold_values)
        #self.preview(background_mask)

        #find contours and select the contour with the greatest area
        if is_cv4():
            im2 = im
            contours, hierarchy = cv2.findContours(background_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            im2, contours, hierarchy = cv2.findContours(background_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        x_min, x_max, y_min, y_max = self.get_boundary_points_from_contour(im2, contours)
            
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

        return ROI

    def remove_small_blobs(self, image_mask, image):

        # self.preview(image_mask)
        # self.preview(image)

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

    def process_image(self, im):
        """ given opencv/numpy image, run the processing method 
        
        return two images - only screws with color and only screws masked
         """
        ROI = self.crop_image_to_box_region(im)
        self.preview(ROI)
        self.count_num_red_screws(ROI)

        only_screws_im, masked_only_screws_im = self.threshold_screw_images(ROI)
        # self.preview(masked_only_screws_im)

        return only_screws_im, masked_only_screws_im

    def count_num_red_screws(self, im):
        red_mask  = self.threshold_red(im)
        num_grey_screws = 0


        blue_mask = self.threshold_color(im, self.blue_threshold_values)

        self.preview(blue_mask)

        num_red_screws = self.count_number_of_screws(red_mask, 500)

        num_blue_screws = self.count_number_of_screws(blue_mask, 100)
        # num_grey_screws = self.count_number_of_screws(grey_mask, 100)

        print(num_blue_screws)
        # self.preview(red_im)

        return num_red_screws, num_blue_screws, num_grey_screws


    def process_image_just_crop(self, im, mask=False):
        """ given opencv/numpy image, run the processing method
         """
        ROI = self.crop_image_to_box_region(im)

        if ROI is None:
            size = [0,0]
        else:
            size = ROI.shape
        desired_size = 800
        delta_w = desired_size - size[1]
        delta_h = desired_size - size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        ROI = cv2.copyMakeBorder(ROI, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)


        # self.preview(ROI)
        # only_screws_im, masked_only_screws_im = self.threshold_screw_images(ROI)
        # self.preview(only_screws_im)

        return ROI

    def process_image_crop_threshold(self, im):
        """ given opencv/numpy image, run the processing method
         """
        ROI = self.crop_image_to_box_region(im)
        ROI = self.threshold_screw_images(ROI)

        return ROI


    def classify_image_contour(self, im):
        """ Given unprocessed image, classify everything"""
        ROI = self.crop_image_to_box_region(im)
        self.preview(ROI)
        num_red_screws = self.count_num_red_screws(ROI)

        return num_red_screws

        # return only_screws_im, masked_only_screws_im



    def count_number_of_screws(self, masked_only_screws_im, threshold_num):

        # self.preview(masked_only_screws_im)
        
        if is_cv4():
            contours, hierarchy = cv2.findContours(masked_only_screws_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            im2, contours, hierarchy = cv2.findContours(masked_only_screws_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        num_contours = 0
        # area = []
        for i in range(len(contours)):
            area_image = cv2.contourArea(contours[i])
            if area_image >= threshold_num:
                num_contours += 1
            # area.append(area_image)
            
        # cont = sorted(contours, key = cv2.contourArea, reverse = True)
        # print(area)


        # num_contours = len(contours)
        # print(num_contours)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(only_screws_im,'Number of screws:' + str(num_contours),(50,50), font, 1,(255,255,255),2,cv2.LINE_AA)
        
        # self.preview(only_screws_im)

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
            #im1 = attempt.process_image_light(im)
            attempt.save_image(im1, f.path)
            #attempt.preview(im1)
    # attempt.count_number_of_screws(im)