# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 15:53:47 2019

@author: Carter
"""

#inference code
# use to predict the class of a new image

import PIL
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2


class Predictor():
    """
    Predictor class takes the .h5 file as input
    """
    
    def __init__(self,model):
        self.model = model
        self.classifier = load_model(model)


    label_dict =  {0: '0bolts0big0small',
                     1:'0bolts0big1small',
                     2:'0bolts0big2small',
                    3: '0bolts1big0small',
                    4: '0bolts1big1small',
                     5:'0bolts1big2small',
                     6: '1bolts0big0small',
                    7: '1bolts0big1small',
                     8: '1bolts0big2small',
                    9: '1bolts1big0small',
                     10: '1bolts1big1small',
                     11: '1bolts1big2small',
                     12:'2bolts0big0small',
                     13:'2bolts0big1small',
                     14:'2bolts0big2small',
                     15:'2bolts1big0small',
                     16:'2bolts1big1small',
                     17:'2bolts1big2small',
                     18:'3bolts0big0small',
                     19:'3bolts0big1small',
                     20:'3bolts0big2small',
                     21:'3bolts1big0small',
                     22:'3bolts1big1small',
                     23:'3bolts1big2small',
                     24:'4bolts2big2small'}
    
    
    def modify_image(self,im):
        im = cv2.resize(im,(128,128))
        im = np.asarray(im, dtype = 'uint8')
        im = im/255
        im = im.reshape((1,128,128,3))
        return im
    
    def show_image(self,image, label):
        image = image.reshape((128,128,3))
        plt.imshow(image)
        plt.title(label)
    
    def predict_image_class(self,image_file, raw_image = False, verbose = 0, dic = label_dict):
        """
        INPUTS
        image file: either path to image file or a raw image shape = (x,y,3) 
        raw_image: True - raw image given False - file path to image given
        verbose: display the image with label after the prediction
        OUTPUTS
        label -string
        lab_number - encoded scalar for each class
        confidence - % confidence of the class
        """
        
        if raw_image==False:
            im = cv2.imread(image_file)
        else:
            im = image_file
            
        im = self.modify_image(im)
        
        classifier = self.classifier
        out = classifier.predict(im)
#        print("out", out)
        
        max_per = np.max(out)
        bin_out = np.where(out == max_per, 1, 0)
        index = np.where(bin_out == 1)
        num_lab = index[1][0]
        label = dic[num_lab]
        
        if verbose:
            self.show_image(im,label)
        
        return label, num_lab, max_per
    
    def predict_image_class_binary(self,image_file, raw_image = False, verbose = 0, dic = label_dict):
        """
        INPUTS
        image file: either path to image file or a raw image shape = (x,y,3) 
        raw_image: True - raw image given False - file path to image given
        verbose: display the image with label after the prediction
        OUTPUTS
        label -string
        lab_number - encoded scalar for each class
        confidence - % confidence of the class
        """
        
        if raw_image==False:
            im = PIL.Image.open(image_file)
        else:
            im = image_file
            
        im = self.modify_image(im)
        
        classifier = self.classifier
        out = classifier.predict(im)
        print("out", out)
        
        max_per = np.max(out)
        if max_per < 0.5:
            num_lab = 0
            label = 'Big Gear Present'
        else:
            num_lab = 1
            label = 'No Big Gear'
#        bin_out = np.where(out == max_per, 1, 0)
#        index = np.where(bin_out == 1)
#        num_lab = index[1][0]
#        label = dic[num_lab]
        
        if verbose:
            self.show_image(im,label)
        
        return label, num_lab, max_per

