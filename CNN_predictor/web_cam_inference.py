# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 21:48:24 2019

@author: Carter
"""

#web_cam_inference
from predict_image_class import Predictor 
from color_threshold_attempt_wrapped import ColorThresholdAttempt
import cv2

new_predictor = Predictor(model = 'crop_20_128x128.h5')
new_attempt = ColorThresholdAttempt()

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    
    if ret:
        im = new_attempt.process_image_just_crop(frame)
        if len(im.shape) == 3:
            label, label_num, confidence  = new_predictor.predict_image_class(im,raw_image = True, verbose=0)
        else:
            label, confidence = "bad image", -1
            
        im_out = cv2.putText(frame, label + " " + str(confidence), (50,50), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255,0,0), 2, cv2.LINE_AA) 
        
        # Display the resulting frame
        cv2.imshow('frame',im_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()