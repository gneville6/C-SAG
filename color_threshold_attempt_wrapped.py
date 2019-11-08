# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:36:06 2019

@author: Carter
"""

#CV project test

import numpy as np
import cv2
import glob 
import os
import scipy
import random
from collections import Counter


class ColorThresholdAttempt:
    def __init__(self):
        self.blue_threshold_values = (np.array([99,50,50]), np.array([115,255,255])) # lower values, upper values
        self.green_threshold_values = (np.array([60,0,0]), np.array([90,255,255]))  # lower values, upper values
        self.white_threshold_values = (np.array([0,0,230]), np.array([255,255,255]))

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
            self.process_image(im)

    def process_image(self, im):
        """ given opencv/numpy image, run the processing method """
        self.preview(im)

    def RANSAC_slope_detection(self, cont_pic, x_vals, y_vals):
        """ compute slopes of boundary detection given image and contour points """
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

    def run(self):
        path = 'bin_images-jpg'


        lis = os.listdir('bin_images-jpg')
        # print(lis)

        for item in lis:
            im = cv2.imread(path + '/' + item)

            reduct = 0.25
            w = int(im.shape[0] * reduct)
            h = int(im.shape[1] * reduct)
            im = cv2.resize(im,(h,w))
            
            self.preview(im)            
            
            #step 1 gaussian blur
            im = cv2.GaussianBlur(im,(7,7), 7)
            
            #step 2 green threshold
            green_outline = self.threshold_color(im, self.green_threshold_values)
            
            #find contours and select the contour with the greatest area
            im2, contours, hierarchy = cv2.findContours(green_outline, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #    cv2.drawContours(im, contours, -1, (0,255,0), 3)
            #    preview(im)
            
            area = []
            for i in range(len(contours)):
                area.append(cv2.contourArea(contours[i]))
                
            cont = sorted(contours, key = cv2.contourArea, reverse = True)
            cont_pic = im.copy()
            cv2.drawContours(cont_pic, cont[0], -1, (0,255,0), 3)
            self.preview(cont_pic)
            
            bucket_cont = np.asarray(cont[0])
            
            #grab the max & min X-Y values of the largest green contour
            x_vals = []
            y_vals = []
            for i in range(len(bucket_cont)):
                x_vals.append(bucket_cont[i][0][0])
                y_vals.append(bucket_cont[i][0][1])            
            

            # self.RANSAC_slope_detection(cont_pic, x_vals, y_vals)

            x_1 = max(x_vals)
            x_max_ind = x_vals.index(max(x_vals))
            y_1 = y_vals[x_max_ind]
            cv2.circle(cont_pic,(y_1,x_1), 20, (0,0,255), -1)
            
            #crop additional space off of the max values
            #figure out how to transform image/homography to overhead view
            cut_pad = 40    
            x_max = max(x_vals) - cut_pad
            x_min = min(x_vals) + cut_pad
            y_max = max(y_vals) - cut_pad
            y_min = min(y_vals) + cut_pad
            
            #crop to the sized of the max values on that contour
            ROI = im[y_min:y_max,x_min:x_max]
            self.preview(ROI)
            
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
            
            green_thresh = self.threshold_color(ROI, self.green_threshold_values)
            #preview(green_thresh)
            green_thresh_inverted = cv2.bitwise_not(green_thresh)
            green_thresh_open_close = self.open_close_image(green_thresh_inverted, 4)
            #green_thresh_inverted = morphOps(green_thresh_inverted, 5)
            
            res = cv2.bitwise_and(ROI,ROI, mask= green_thresh_open_close)
            #preview(res)
            
            
            im2, contours, hierarchy = cv2.findContours(green_thresh_open_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            num_contours = len(contours)
            print(num_contours)
            font = cv2.FONT_HERSHEY_SIMPLEX
            #    cv2.putText(res,'Number of screws:' + str(num_contours),(50,50), font, 1,(255,255,255),2,cv2.LINE_AA)
            self.preview(res)
            # TODO: what should the return be
            return num_contours

if __name__ == '__main__':
    attempt = ColorThresholdAttempt()
    attempt.run()
    path = 'bin_images-jpg'
    # attempt.process_images_from_folder(path)