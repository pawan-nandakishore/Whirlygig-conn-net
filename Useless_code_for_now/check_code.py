# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:36:27 2017

@author: pawan
"""

import numpy as np
from skimage.color import gray2rgb
from skimage.io import imsave
import cv2
import matplotlib.pyplot as plt

def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)
    
    
    

raw_image = cv2.imread('raw_boundaries3.png',0)

image_list= []
        
        # the number of empty columns are 
window =10        
window =window
        
plt.imshow(raw_image)
        
#        
final_index =raw_image.shape[1]*raw_image.shape[0]
indices = np.array(range(0,final_index))

[x1,y1] = ind2sub(raw_image.shape,indices) 
grt_indx_x =  np.where(x1>window)  
grt_indx_y  =  np.where(y1>window)
grt_indx = np.intersect1d(grt_indx_x, grt_indx_y)
          
all_pixel_list_array = np.array([x1,y1]).T    
all_pixel_list_array = all_pixel_list_array[grt_indx,:]

final_pixel_list_array =all_pixel_list_array
           
    
        
        
for pixel in range(0,len(final_pixel_list_array)):
         
 pixelx = final_pixel_list_array[pixel,0]
 pixely = final_pixel_list_array[pixel,1]

  
 cropped_image = raw_image[pixelx-window:pixelx+window,pixely-window:pixely+window] 
        
 image_list.append(cropped_image)     


        
         