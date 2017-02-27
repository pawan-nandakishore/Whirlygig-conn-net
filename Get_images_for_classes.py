# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 23:42:06 2017

@author: pawan
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import cv2 





#Function version 

#input 

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
    
#this function has two flags one is the background pixel value that you have to input, and the other is the 
#discard back, if discard back is 1 it takes into account only the background pixels that dont have a common value
# this is benificial in the case that you have a large background value and very few foreground pixels, or if you have
# a situation where one class dominates over the other
    
    
    
    
def get_images_per_class(labeled_image,raw_image,background_pixel_val,discard_back): 
        if not background_pixel_val: 
             background_pixel_val =0 
        red_pixel_class = labeled_image
        
        list_of_classes =  np.unique(red_pixel_class)
        
        image_list= [[] for x in range(0,len(list_of_classes))]
        
        # the number of empty columns are 
        
        window =10 
        
        plt.imshow(raw_image)
        
        list_indx= 0 
        for i in list_of_classes: 
           
           class_pixel_list = np.where(red_pixel_class == i )
          
           grt_indx_x =  np.where(class_pixel_list[0]>window)  
           grt_indx_y  =  np.where(class_pixel_list[1]>window)
           grt_indx = np.intersect1d(grt_indx_x, grt_indx_y)
          
           class_pixel_list_array = np.asanyarray(class_pixel_list).T
           
           class_pixel_list2 = class_pixel_list_array[grt_indx,:]
           
           class_pixel_list = [class_pixel_list2[:,0],class_pixel_list2[:,1]]
           
           class_pixel_list_array = np.asanyarray(class_pixel_list).T
           
           final_pixel_list_array  = class_pixel_list_array

           # this section is if you want to exclude majority of the pixels from the 
           #back ground. 
           if discard_back ==1: 
               [grt_zero_x, grt_zero_y ] = np.where(raw_image>background_pixel_val)
               grt_zero = np.array([grt_zero_x,grt_zero_y]).T 
               
               
               grt_zero_ind = sub2ind(raw_image.shape, grt_zero[:,0],grt_zero[:,1]) 
               class_pixel_list_ind = sub2ind(raw_image.shape, class_pixel_list_array[:,0],class_pixel_list_array[:,1]) 
               
               grt_zero_pixlist_intersect = np.intersect1d(grt_zero_ind,class_pixel_list_ind)
               
               grt_zero_list = ind2sub(raw_image.shape,grt_zero_pixlist_intersect)
               grt_zero_array= np.array([grt_zero_list[0],grt_zero_list[1]]).T
               final_pixel_list_array = grt_zero_array
           
            
             
      
           for pixel in range(0,len(final_pixel_list_array)):
         
             pixelx = final_pixel_list_array[pixel,0]
             pixely = final_pixel_list_array[pixel,1]
    
              
             cropped_image = raw_image[pixelx-window:pixelx+window,pixely-window:pixely+window] 
                    
             image_list[list_indx].append(cropped_image)     
        
            
        
        
        
           list_indx +=1 
        return(image_list)
        
        
        


red_pixel_class = np.load('red_pixel_class.npy')
raw_image = cv2.imread('raw_boundaries3.png',0)


imagelist = get_images_per_class(red_pixel_class,raw_image,0,1)
filename= 'list_of_images_for_all_classes'

np.save(filename,imagelist)



