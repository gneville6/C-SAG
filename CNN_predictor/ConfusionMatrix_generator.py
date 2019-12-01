# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 17:15:24 2019

@author: Carter
"""

#confusion matrix for test set 
#find out where it is messing up

from predict_image_class import Predictor 
import os
import numpy as np

#get the test set
from keras.preprocessing.image import ImageDataGenerator
file = 'C:/Users/Carter/Desktop/Computer Vision/Group_project/Minimal Preprocess-just crop/Minimal Preprocess-just crop/Test_set/'

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(file,
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'categorical')

classes = test_set.classes
image_paths = test_set.filepaths

#create new class instance
new_predictor = Predictor(model = 'Min_pre_ep_35_128x128.h5')

#create confusion matrix

cm = np.zeros((25,25))
right = 0
wrong = 0
wrong_record = []
for i in range(len(image_paths)):
    file = image_paths[i]
    result, lab_num, confidence = new_predictor.predict_image_class(file,verbose=0)
    cm[lab_num][classes[i]] += 1
    if lab_num == classes[i]:
        right +=1
    else:
        wrong +=1
        wrong_record.append((i,result,lab_num,confidence))