# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:14:49 2019

@author: Carter
"""

#example classification


from predict_image_class import Predictor 
import PIL
import os

#sample directory
directory = '/Users\Carter\Desktop\Computer Vision\Group_project\Minimal Preprocess-just crop\Minimal Preprocess-just crop/Test_set/1bolts0big2small/'
lis = os.listdir(directory)
file = directory + lis[22]

#create new class instance
new_predictor = Predictor(model = 'Min_pre_ep_35_128x128.h5')

#example with image input file path
label, label_num, confidence = new_predictor.predict_image_class(file,raw_image = False, verbose=0)
    
#example with raw image
image = PIL.Image.open(file)
label, label_num, confidence  = new_predictor.predict_image_class(image,raw_image = True, verbose=1)