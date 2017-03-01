# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 19:26:15 2017

@author: pawan
"""

import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import Get_images_for_classes_functions as gi 



marked_image = cv2.imread('marked_image')
raw_image = cv2.imread('raw_image2')

marked_image = cv2.resize(marked_image,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
raw_image = cv2.resize(raw_image,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)

marked_image2 = marked_image.copy()
cut_off_percent =15.0 

cut_off_vals = [marked_image.shape[0]*(cut_off_percent/100), marked_image.shape[1]*(cut_off_percent/100)] 

end_vals = [marked_image.shape[0]-cut_off_vals[0],marked_image.shape[1]-cut_off_vals[1]] 

end_vals= np.round(end_vals) 
marked_image =marked_image[cut_off_vals[0]:end_vals[0],cut_off_vals[1]:end_vals[1]  ]
raw_image =raw_image[cut_off_vals[0]:end_vals[0],cut_off_vals[1]:end_vals[1]  ]

mask = np.zeros(marked_image.shape)
mask = mask[:,:,1]

marked_pixels= np.where(marked_image[:,:,0] != marked_image[:,:,2])
mask[marked_pixels[0],marked_pixels[1] ] = 255

marked_pixels = np.array([marked_pixels[0],marked_pixels[1]])


plt.figure
plt.imshow(raw_image)

filename= 'labeled_image' 
#
np.save(filename,mask)
filename= 'raw_image_cropped' 
plt.imsave(filename, raw_image)






#
#markingLower = np.array([0 ,  0 , 230 ],dtype=np.uint8)
#markingUpper = np.array([0,  0 , 255],dtype=np.uint8)