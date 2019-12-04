import numpy as np
import cv2
import glob 
import os
import scipy
import random
from collections import Counter
from PIL import Image, ImageOps
from math import sqrt
import scipy.stats as stats
from scipy.stats import skewnorm
import matplotlib.pyplot as plt
import json
import copy

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


class Image:
    RED_THRESHOLD_LOWER_1 = np.array([0, 0, 0])
    RED_THRESHOLD_UPPER_1 = np.array([12, 255, 255])
    RED_THRESHOLD_LOWER_2 = np.array([150, 0, 0])
    RED_THRESHOLD_UPPER_2 = np.array([179, 255, 255])

    BLUE_THRESHOLD_LOWER = np.array([95, 0, 0])
    BLUE_THRESHOLD_UPPER = np.array([150, 255, 255])

    GREEN_THRESHOLD_LOWER = np.array([50, 0, 0])
    GREEN_THRESHOLD_UPPER = np.array([80, 255, 255])

    WHITE_THRESHOLD_LOWER = np.array([220, 220, 220])
    WHITE_THRESHOLD_UPPER = np.array([255, 255, 255])

    def __init__(self, label, image_name, image):
        self.label = label
        self.num_red_gears = int(label[6])
        self.num_blue_gears = int(label[10])
        self.num_gray_bolts = int(label[0])

        self.image_name = image_name
        self.image = image
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        self.green_threshold_values = (np.array([60,0,0]), np.array([90,255,255]))  # lower values, upper values


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

    def crop_image_to_box_region(self, mask=True):
        
        #step 2 green threshold for box
        background_mask = self.threshold_color(self.image, self.green_threshold_values)
        #self.preview(background_mask)

        #find contours and select the contour with the greatest area
        if is_cv4():
            im2 = self.image
            contours, hierarchy = cv2.findContours(background_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            im2, contours, hierarchy = cv2.findContours(background_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        x_min, x_max, y_min, y_max = self.get_boundary_points_from_contour(im2, contours)
            
        #crop to the sized of the max values on that contour
        ROI = self.image[y_min:y_max,x_min:x_max]
        
        if mask:
            mask = np.zeros(self.image.shape).astype(self.image.dtype)
            cont = sorted(contours, key = cv2.contourArea, reverse = True)
            cv2.fillPoly(mask, [cont[0]], [255, 255, 255])
            mask_inverted = cv2.bitwise_not(mask)
            mask_inverted = mask_inverted[y_min:y_max,x_min:x_max]
            mask_inverted[mask_inverted[:,:, 0] > 128] = [0, 255, 0]
            mask = mask[y_min:y_max, x_min:x_max] + mask_inverted
            ROI = cv2.bitwise_and(ROI, mask)
            #self.preview(ROI)

        self.image = ROI

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

    def preview(self):
        cv2.imshow("preview", self.image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def set_pixels(self, red, blue, gray):
        self.red_pixels = red
        self.blue_pixels = blue
        self.gray_pixels = gray

    def super_saturate(self):
        #self.hsv[:, :, 1] = 255
        self.hsv[:, :, 2] = 255
        self.image = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)

    def remove_outer(self):
        kernel_size = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        green_mask = cv2.inRange(self.hsv, self.GREEN_THRESHOLD_LOWER, self.GREEN_THRESHOLD_UPPER)
        green_mask = cv2.erode(green_mask, kernel=kernel)
        green_mask = cv2.dilate(green_mask, kernel=kernel)
        
        contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        boundary = contours[max_index]
        stencil = np.zeros(self.hsv.shape).astype(self.hsv.dtype)
        cv2.fillPoly(stencil, [boundary], [255, 255, 255])
        self.hsv = cv2.bitwise_and(self.hsv, stencil)
        self.image = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)

    def fix_glare(self):
        h, s, v = cv2.split(self.hsv)
        non_sat = s < 180
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        non_sat = cv2.erode(non_sat.astype(np.uint8), disk)
        v2 = v.copy()
        v2[non_sat == 0] = 0
        glare = v2 > 200
        glare = cv2.dilate(glare.astype(np.uint8), disk, iterations=2)
        self.image = cv2.inpaint(self.image, glare, 5, cv2.INPAINT_NS)
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

    def remove_glare(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        self.hsv[thresh == 255] = 0
        
        kernel_size = 10
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        self.hsv = cv2.erode(self.hsv, kernel=kernel)
        self.image = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)

    def remove_green(self):
        green_mask = cv2.inRange(self.hsv, self.GREEN_THRESHOLD_LOWER, self.GREEN_THRESHOLD_UPPER)
        
        kernel_size = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        green_mask = cv2.dilate(green_mask, kernel=kernel)
        green_mask = cv2.erode(green_mask, kernel=kernel)

        self.hsv = cv2.bitwise_and(self.hsv, self.hsv, mask=cv2.bitwise_not(green_mask))
        self.image = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)

    def solidify(self):
        for i in range(20):
            self.hsv = cv2.medianBlur(self.hsv, 5)
        self.image = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)

    def count_white(self, remove=True):
        white_mask = cv2.inRange(self.image, self.WHITE_THRESHOLD_LOWER, self.WHITE_THRESHOLD_UPPER)

        kernel_size = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        red_mask = cv2.dilate(white_mask, kernel=kernel)
        red_mask = cv2.erode(white_mask, kernel=kernel)

        if remove:
            self.hsv = cv2.bitwise_and(self.hsv, self.hsv, mask=cv2.bitwise_not(white_mask))
            self.image = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)
        return cv2.countNonZero(white_mask)

    def count_red(self, remove=True):
        red_mask1 = cv2.inRange(self.hsv, self.RED_THRESHOLD_LOWER_1, self.RED_THRESHOLD_UPPER_1)
        red_mask2 = cv2.inRange(self.hsv, self.RED_THRESHOLD_LOWER_2, self.RED_THRESHOLD_UPPER_2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        kernel_size = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        red_mask = cv2.dilate(red_mask, kernel=kernel)
        red_mask = cv2.erode(red_mask, kernel=kernel)

        if remove:
            self.hsv = cv2.bitwise_and(self.hsv, self.hsv, mask=cv2.bitwise_not(red_mask))
            self.image = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)
        return cv2.countNonZero(red_mask)

    def count_blue(self, remove=True):
        blue_mask = cv2.inRange(self.hsv, self.BLUE_THRESHOLD_LOWER, self.BLUE_THRESHOLD_UPPER)
        
        kernel_size = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        blue_mask = cv2.dilate(blue_mask, kernel=kernel)
        blue_mask = cv2.erode(blue_mask, kernel=kernel)

        if remove:
            self.hsv = cv2.bitwise_and(self.hsv, self.hsv, mask=cv2.bitwise_not(blue_mask))
            self.image = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)
        return cv2.countNonZero(blue_mask)

    def count_what_is_left(self):
        all_mask = cv2.inRange(self.hsv, np.array([0, 0, 0]), np.array([179, 255, 255]))
        return cv2.countNonZero(all_mask)


class ProbabilityDistribution:
    def __init__(self, truth):
        self.truth = truth
        self.pixels = []
        self.pdf = None

    def add(self, value):
        self.pixels.append(value)

    def create(self):
        def __create_pdf(val):
            a, loc, scale = skewnorm.fit(val)
            return skewnorm(a, loc, scale)

        self.pdf = __create_pdf(self.pixels)

    def predict(self, value):
        return self.pdf.pdf(value)

    def average(self):
        if len(self.pixels) > 0:
             sum(self.pixels) / len(self.pixels)
        return 0

    def std(self):
        mu = average()
        if len(self.pixels) > 0:
            return sqrt(sum(pow(sample - mu, 2) for sample in self.pixels) / (len(self.pixels) - 1))
        return 0

class ConfusionMatrix:
    def __init__(self, n):
        self.matrix = []
        for i in range(n):
            self.matrix.append([])
            for j in range(n):
                self.matrix[-1].append(0)
        self.matrix = np.array(self.matrix)
        self.total = 0

    def increment(self, prediction, truth):
        self.matrix[prediction][truth] += 1
        self.total += 1

    def accuracy(self):
        n = len(self.matrix)
        num = 0
        for i in range(n):
            num += self.matrix[i][i]
        return num / self.total

    def display(self):
        classes = np.arange(self.matrix.shape[0])
        cmap = plt.cm.Blues
        fig, ax = plt.subplots()
        normalize = False
        im = ax.imshow(self.matrix, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(self.matrix.shape[1]),
               yticks=np.arange(self.matrix.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
 
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
 
        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        thresh = self.matrix.max() / 2.
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                ax.text(j, i, format(self.matrix[i, j], fmt),
                        ha="center", va="center",
                        color="white" if self.matrix[i, j] > thresh else "black")
        fig.tight_layout()
        plt.show()

def read_images_in_path(folder):
    if not os.path.exists(folder):
        print('Error - The folder does not exist')
        return

    images = []

    for subfolder in os.scandir(folder):
        if not subfolder.is_dir():
            print('Error - Subfolder is not a directory')
            return
        label = subfolder.name
        for image_file in os.scandir(subfolder.path):
            images.append(Image(label, image_file.name, cv2.imread(image_file.path)))
    return images

def read_images_in_single_folder(subfolder):
    images = []
    label = '2bolts0big2small'
    for image_file in os.scandir(subfolder):
        images.append(Image(label, image_file.name, cv2.imread(image_file.path)))
    return images  

def plot_pdfs(pdfs):
    averages = [pdf.average() for pdf in pdfs]
    stds = [pdf.std() for pdf in pdfs]

    x_min = max(min(averages) - 3 * max(stds), 0)
    x_max = max(avg) + 3 * max(std)
    x = np.linspace(x_min, x_max)
    color = ['red', 'green', 'blue', 'yellow']
    for i, pdf in enumerate(pdfs):
        plt.plot(x, pdf.predict(x), color=color[i], lw=2, label='frozen pdf')
    plt.show()

if __name__ == '__main__':
    debug = True

    probability_distributions = {
        'red': [ProbabilityDistribution(0), ProbabilityDistribution(1)],
        'blue': [ProbabilityDistribution(0), ProbabilityDistribution(1), ProbabilityDistribution(2)],
        'gray': [ProbabilityDistribution(0), ProbabilityDistribution(1), ProbabilityDistribution(2), ProbabilityDistribution(3)]
    }    

    images = read_images_in_single_folder('/home/sean/Images/Test_set/2bolts0big2small')

    print('Computing probability distributions...')
    for image in images:
        print('\t{0:s}/{1:s}'.format(image.label, image.image_name))
        if debug:
            debug_images = {}
            debug_images['original'] = image.image


        image.crop_image_to_box_region()
        image.preview()


        image.fix_glare()
        image.super_saturate()
        if debug:
            debug_images['super_saturate'] = image.image
        #if debug:
        #    debug_images['no_glare'] = image.image
        image.remove_outer()
        if debug:
            debug_images['no_boundary'] = image.image
        image.remove_green()
        image.solidify()
        if debug:
            debug_images['no_green'] = image.image
        gray = image.count_white()
        if debug:
            debug_images['no_white'] = image.image
        red = image.count_red()
        if debug:
            debug_images['no_red'] = image.image
        blue = image.count_blue()
        if debug:
            debug_images['no_blue'] = image.image

        if debug:
            debug_image = np.hstack((np.vstack((debug_images['original'], debug_images['no_boundary'])),
                                     np.vstack((debug_images['no_green'], debug_images['no_white'])),
                                     np.vstack((debug_images['no_red'], debug_images['no_blue']))))
            cv2.imshow('{0:s}/{1:s}'.format(image.label, image.image_name), debug_image)
            cv2.waitKey()

        image.set_pixels(red, blue, gray)
        probability_distributions['red'][image.num_red_gears].add(red)
        probability_distributions['blue'][image.num_blue_gears].add(blue)
        probability_distributions['gray'][image.num_gray_bolts].add(gray)

    for color, prob_list in probability_distributions.items():
        for pd in prob_list:
            pd.create()

    red_cm = ConfusionMatrix(2)
    blue_cm = ConfusionMatrix(3)
    gray_cm = ConfusionMatrix(4)

    print('Computing accuracies...')
    for image in images:
        print('\t{0:s}/{1:s}'.format(image.label, image.image_name))
        red_probabilities = [
            pd.predict(image.red_pixels) for pd in probability_distributions['red']
        ]
        red_prediction = np.argmax(red_probabilities)
        red_cm.increment(red_prediction, image.num_red_gears)

        blue_probabilities = [
            pd.predict(image.blue_pixels) for pd in probability_distributions['blue']
        ]
        blue_prediction = np.argmax(blue_probabilities)
        blue_cm.increment(blue_prediction, image.num_blue_gears)

        gray_probabilities = [
            pd.predict(image.gray_pixels) for pd in probability_distributions['gray']
        ]
        gray_prediction = np.argmax(gray_probabilities)
        gray_cm.increment(gray_prediction, image.num_gray_bolts)

    #red_cm.display()
    print('Red Accuracy: {0:0.2f}'.format(red_cm.accuracy()))

    #blue_cm.display()
    print('Blue Accuracy: {0:0.2f}'.format(blue_cm.accuracy()))

    #gray_cm.display()
    print('Gray Accuracy: {0:0.2f}'.format(gray_cm.accuracy()))