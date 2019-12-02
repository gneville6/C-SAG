#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:36:06 2019

@author: Carter
"""

# CV project test

import numpy as np
import cv2
from cv2 import SimpleBlobDetector
import glob
import os
import scipy
import random
from collections import Counter
from PIL import Image, ImageOps


class Histogram:
    def __init__(self):
        self.test = 4
        self.blue_threshold_values = (np.array([95, 80, 80]), np.array([150, 255, 255]))  # lower values, upper values
        self.red_threshold_values = (np.array([150, 160, 70]), np.array([255, 255, 255]))  # lower values, upper values
        self.red_threshold_values2 = (np.array([0, 160, 100]), np.array([12, 255, 255]))
        self.green_threshold_values = (np.array([40, 1, 1]), np.array([90, 255, 255]))  # lower values, upper values
        self.white_threshold_values = (np.array([0, 0, 0]), np.array([180, 200, 200]))
        # self.gray_threshold_values = (np.array([0, 5, 100]), np.array([255, 40, 170]))

        self.params = cv2.SimpleBlobDetector_Params()



        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        params.minDistBetweenBlobs = 0
        # Change thresholds
        params.minThreshold = 0;
        params.maxThreshold = 200;

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 100
        params.maxArea = 6000

        # Filter by Circularity
        params.filterByCircularity = False
        params.minCircularity = 0.1

        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.87

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01
        params.maxInertiaRatio = 0.3

        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            self.detector = cv2.SimpleBlobDetector(params)
        else:
            self.detector = cv2.SimpleBlobDetector_create(params)


    def preview(self, im):
        cv2.imshow("preview", im)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def save_image(self, image, name):
        cv2.imwrite(name, image)

    def threshold_color(self, im, threshold_values):

        # Convert BGR to HSV
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        # define range colors in HSV
        lower_threshold_value = threshold_values[0]
        upper_threshold_value = threshold_values[1]
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_threshold_value, upper_threshold_value)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(im, im, mask=mask)
        # self.preview(im)
        # self.preview(mask)
        return mask

    def resize_image(self, im, scale):
        """
        scales image
        """
        w = int(im.shape[0] * scale)
        h = int(im.shape[1] * scale)
        im = cv2.resize(im, (h, w))
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
            # self.preview(im2)

    def fill_in_glare(self, ROI):
        """ WIP currently unused function """
        green_mask = np.zeros(ROI.shape)
        green_mask[:, :, 0] = 32
        green_mask[:, :, 1] = 100
        green_mask[:, :, 2] = 1
        green = np.asarray(green_mask, dtype='uint8')
        white = self.threshold_color(ROI, self.white_threshold_values)
        result = cv2.bitwise_and(green, ROI, mask=white)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if sum(result[i, j, :]) == 0:
                    result[i, j, :] = [255, 255, 255]
        result2 = cv2.bitwise_and(result, ROI)
        # blue_thresh = threshold_blue(ROI)
        # preview(blue_thresh)
        return result2

    def get_boundary_points_from_contour(self, im, contours):
        area = []
        for i in range(len(contours)):
            area.append(cv2.contourArea(contours[i]))

        cont = sorted(contours, key=cv2.contourArea, reverse=True)
        cont_pic = im.copy()
        cv2.drawContours(cont_pic, cont[0], -1, (0, 255, 0), 3)
        # self.preview(cont_pic)

        bucket_cont = np.asarray(cont[0])

        # grab the max & min X-Y values of the largest green contour
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

        # crop additional space off of the max values
        # figure out how to transform image/homography to overhead view
        cut_pad = 40
        x_max = max(x_vals) - cut_pad
        x_min = min(x_vals) + cut_pad
        y_max = max(y_vals) - cut_pad
        y_min = min(y_vals) + cut_pad

        return x_min, x_max, y_min, y_max

    def crop_image_to_box_region(self, im, mask=True):
        ''' input raw image, return region of image that has the box inside '''
        # step 1 gaussian blur
        if self.test == 0:
            im = cv2.GaussianBlur(im, (7, 7), 7)

        # step 2 green threshold for box
        background_mask = self.threshold_color(im, self.green_threshold_values)
        # self.preview(background_mask)

        # find contours and select the contour with the greatest area
        im2, contours, hierarchy = cv2.findContours(background_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            x_min, x_max, y_min, y_max = self.get_boundary_points_from_contour(im2, contours)
            # crop to the sized of the max values on that contour
            ROI = im[y_min:y_max, x_min:x_max]

            if mask:
                mask = np.zeros(im.shape).astype(im.dtype)
                cont = sorted(contours, key=cv2.contourArea, reverse=True)
                cv2.fillPoly(mask, [cont[0]], [255, 255, 255])
                mask_inverted = cv2.bitwise_not(mask)
                mask_inverted = mask_inverted[y_min:y_max, x_min:x_max]
                mask_inverted[mask_inverted[:, :, 0] > 128] = [0, 255, 0]
                mask = mask[y_min:y_max, x_min:x_max] + mask_inverted
                ROI = cv2.bitwise_and(ROI, mask)
                # self.preview(ROI)
        else:
            ROI = im
        return ROI

    def process_image(self, im):
        """ given opencv/numpy image, run the processing method

        return two images - only screws with color and only screws masked
         """
        ROI = self.crop_image_to_box_region(im)
        # self.preview(ROI)
        only_screws_im, masked_only_screws_im = self.threshold_screw_images(ROI)
        # self.preview(only_screws_im)

        return only_screws_im, masked_only_screws_im

    def process_image_just_crop(self, im, mask=False):
        """ given opencv/numpy image, run the processing method

        return two images - only screws with color and only screws masked
         """
        ROI = self.crop_image_to_box_region(im)

        if ROI is None:
            ROI = np.zeros([800, 800, 3])

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
        # only_screws_im, masked_only_screws_im = self.threshold_screw_images(ROI)
        # self.preview(only_screws_im)

        return ROI

    def process_image_just_crop(self, im, mask=False):
        """ given opencv/numpy image, run the processing method

        return two images - only screws with color and only screws masked
         """
        ROI = self.crop_image_to_box_region(im)

        if ROI is None:
            ROI = np.zeros([800, 800, 3])

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
        # only_screws_im, masked_only_screws_im = self.threshold_screw_images(ROI)
        # self.preview(only_screws_im)

        return ROI

    def threshold_hist(self, colorHist):
        print(colorHist)
        # threshold = [410000, 508755, 260310, 1203600,503600] 937380
        threshold = [410000, 708755, 500310, 11503600, 700000, 200000]
        classified = [0, 0, 0]
        if colorHist[1] > threshold[0]:
            classified[1] = 1
        if colorHist[0] > threshold[1]:
            classified[0] = 2
        elif colorHist[0] > threshold[2]:
            classified[0] = 1
        if colorHist[2] > threshold[3]:
            classified[2] = 3
        elif colorHist[2] > threshold[4]:
            classified[2] = 2
        elif colorHist[2] > threshold[5]:
            classified[2] = 1
        return classified

    def crop_histogram(self, im, mask=False):
        """ given opencv/numpy image, run the processing method
        return two images - only screws with color and only screws masked
         """

        ROI = self.threshold_color(im, self.red_threshold_values)
        ROI2 = self.threshold_color(im, self.red_threshold_values2)
        ROI = cv2.add(ROI, ROI2)
        kernel = np.ones([3, 3]) / 9
        ROI = cv2.morphologyEx(ROI, cv2.MORPH_OPEN, kernel)
        sumRed = np.sum(ROI)
        ROI_red = ROI
        # print("Red= " + str(sumRed))

        ROI = cv2.bitwise_and(im, cv2.cvtColor(ROI_red, cv2.COLOR_GRAY2RGB))
        ROI = cv2.bitwise_not(ROI)
        ROI_min_red = cv2.bitwise_and(ROI, im)

        ROI = ROI_min_red  # self.crop_image_to_box_region(im, False)
        ROI = self.threshold_color(ROI, self.blue_threshold_values)
        kernel = np.ones([2, 2]) / 4
        ROI = cv2.morphologyEx(ROI, cv2.MORPH_OPEN, kernel)
        ROI = cv2.dilate(ROI, kernel, iterations=2)
        ROI_blue = ROI
        sumBlue = np.sum(ROI)
        #print("Blue= " + str(sumBlue))

        ROI = cv2.bitwise_and(ROI_min_red, cv2.cvtColor(ROI_blue, cv2.COLOR_GRAY2RGB))
        ROI = cv2.bitwise_not(ROI)
        ROI_min_blue = cv2.bitwise_and(ROI, ROI_min_red)

        ROI_min_blue = self.crop_image_to_box_region(ROI_min_blue, mask=True)

        im2 = np.mean(ROI_min_blue, axis=2)
        im2 = [im2, im2, im2]
        im2 = np.stack(im2, axis=2)
        ROI_min_blue[im2 > 230] = 0
        im2 = np.subtract(ROI_min_blue, im2)
        im2 = np.abs(im2)
        im1 = np.mean(im2, axis=2)
        im2 = [im1,im1,im1]
        im2 = np.stack(im2, axis=2)
        ROI_min_blue[im2 > 12] = 0
        ROI_min_blue_mask = self.threshold_color(ROI_min_blue, self.white_threshold_values)
        ROI_min_blue = cv2.bitwise_and(ROI_min_blue, cv2.cvtColor(ROI_min_blue_mask, cv2.COLOR_GRAY2RGB))
        ROI_gray = cv2.cvtColor(ROI_min_blue, cv2.COLOR_RGB2GRAY)

        sumGray = np.sum(ROI_gray)
        print("Gray= " + str(sumGray))

        colorHist = [sumBlue, sumRed, sumGray]
        classify = self.threshold_hist(colorHist)

        return classify


if __name__ == '__main__':
    attempt = Histogram()
    # attempt.run()
    path = 'processed/Test_set'
    numCorrect = 0
    numIncorrect = 0
    # attempt.process_images_from_folder(path)
    for j in os.scandir(path):
        path1 = j.path
        print(path1)
        for f in os.scandir(path1):
            if not f.is_file():
                # wtf
                continue
            im = cv2.imread(f.path)
            classified = attempt.crop_histogram(im)
            correct = True
            '''
            print(f.path[19])
            print(f.path[25])
            print(f.path[29])

            print(f.path[23])
            print(f.path[29])
            print(f.path[33])
            '''

            if int(f.path[19]) - classified[2]  == 0:
                correct = True
                numCorrect += 1
            else:
                correct = False
                numIncorrect += 1
                # attempt.preview(im)

            if correct:
                print(str(f.path) + "Label= " + str(classified[2]) + str("= Correct"))
            else:
                print(str(f.path) + "Label= " + str(classified[2]) + str("= Incorrect"))

    print("Accuracy= " + str(numCorrect / (numCorrect + numIncorrect)))

    # attempt.count_number_of_screws(im)
