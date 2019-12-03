# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 20:34:14 2019

@author: Carter
"""

import os
import cv2

#script to delete certain images that I do not want in the training set

directory = 'crop_w_pad_4_classes_bolts/Training_set/2_bolts/'

lis = os.listdir(directory)

for item in lis:
    file = directory + item
    im = cv2.imread(file)
    cv2.imshow('preview', im)
    if cv2.waitKey() & 0xFF == ord('q'):
        os.remove(file)
        print("removed file", file)
        
cv2.destroyAllWindows()
