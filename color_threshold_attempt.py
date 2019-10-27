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

GREEN_LOWER = np.array([60,0,0])
GREEN_UPPER = np.array([90,255,255])
BLUE_LOWER =  np.array([99,50,50])
BLUE_UPPER = np.array([115,255,255])

def preview(im):
    cv2.imshow("preview", im)
    cv2.waitKey()
    cv2.destroyAllWindows()

def threshold_color(im, lower_bound, upper_bound):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(im,im, mask= mask)
    return mask

def open_close_image(binary_matrix, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erosion = cv2.erode(binary_matrix,kernel,iterations = 1)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    return erosion

def get_bucket_points(green_outline):
    ''' returns edges of green bucket '''
    cut_pad = 40 
    im2, contours, hierarchy = cv2.findContours(green_outline, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(im, contours, -1, (0,255,0), 3)
    # preview(im)
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
        
    cont = sorted(contours, key = cv2.contourArea, reverse = True)
    # cv2.drawContours(im, cont[0], -1, (0,255,0), 3)
    # preview(im)

    bucket_cont = np.asarray(cont[0])

    x_vals = []
    y_vals = []
    for i in range(len(bucket_cont)):
        x_vals.append(bucket_cont[i][0][0])
        y_vals.append(bucket_cont[i][0][1])
       
    x_max = max(x_vals) - cut_pad
    x_min = min(x_vals) + cut_pad
    y_max = max(y_vals) - cut_pad
    y_min = min(y_vals) + cut_pad

    return y_min, y_max, x_min, x_max

def get_screw_contours(ROI):
    ''' given bucket region, return screw contours '''
    green_thresh = threshold_color(ROI, GREEN_LOWER, GREEN_UPPER)
    green_thresh_inverted = cv2.bitwise_not(green_thresh)
    green_thresh_open_close = open_close_image(green_thresh_inverted, 8)

    res = cv2.bitwise_and(ROI,ROI, mask= green_thresh_open_close)

    im2, contours, hierarchy = cv2.findContours(green_thresh_open_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return res, contours

def count_screws(im):
    ''' takes images and returns number of screws '''
    reduct = 0.25
    w = int(im.shape[0] * reduct)
    h = int(im.shape[1] * reduct)
    im = cv2.resize(im,(h,w))


    green_outline = threshold_color(im, GREEN_LOWER, GREEN_UPPER)
    y_min, y_max, x_min, x_max = get_bucket_points(green_outline)

    ROI = im[y_min:y_max,x_min:x_max]

    res, contours = get_screw_contours(ROI)

    num_contours = len(contours)
    print(num_contours)
    preview(res)

    return num_contours


path = 'bin_images-jpg'
lis = os.listdir('bin_images-jpg')
image = cv2.imread(path + '/' + lis[2])
count_screws(image)