# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 19:39:29 2017

@author: pawan
"""

import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import Get_images_for_classes_functions as gi 
import random
from skimage.transform import rotate
from PIL import Image

#labeled_image = np.load('labeled_image.npy')
#raw_image = cv2.imread('raw_image_cropped.png',0)
plt.close('all')
labeled_image_list = np.load('labeled_images.npy')

raw_images_list = np.load('raw_images_cropped.npy')

labeled_image_list = list(labeled_image_list)
raw_images_list = list(raw_images_list)



number_of_classes = 2
window_size = 6
discard_back =1
background_pixel_val = 0 
window= window_size
#marked_cropped_image_list = gi.get_images_per_class(labeled_image,raw_image,0, 1,window_size)
all_marked_cropped_image_list =[[] for i in range(0,number_of_classes)]



for i in range(0,len(labeled_image_list)): 
   single_labeled_image =  labeled_image_list[i]
   single_raw_image = raw_images_list[i]
   class_pixels = np.where(single_labeled_image>0) 
  
 
 
 
 
 
   [marked_cropped_image_list,final_pixel_list] =  gi.get_images_per_class(single_labeled_image,single_raw_image,background_pixel_val, discard_back,window_size)
       
   all_marked_cropped_image_list[0] += marked_cropped_image_list[0]
   all_marked_cropped_image_list[1] += marked_cropped_image_list[1]
   


       
       
       
flipped_marked_cropped_list_lr = []
flipped_marked_cropped_list_ud = [] 
flipped_marked_cropped_list_90 = []
for i in all_marked_cropped_image_list[1]: 
    flipped_marked_cropped_list_lr.append(np.fliplr(i))
    flipped_marked_cropped_list_ud.append(np.flipud(i))
    flipped_marked_cropped_list_90.append(np.rot90(i))



all_marked_cropped_image_list[1] =all_marked_cropped_image_list[1]+ flipped_marked_cropped_list_90+flipped_marked_cropped_list_lr+flipped_marked_cropped_list_ud




marked_cropped_image_list =all_marked_cropped_image_list

percent_of_bck = 60.0
percent_of_bck = percent_of_bck/100.0 

background_images= marked_cropped_image_list[0]
num_background =np.int(len(marked_cropped_image_list[1])+np.round(len(marked_cropped_image_list[0])*percent_of_bck) )
background_images_subset = random.sample(background_images,num_background)



training_data = background_images_subset +marked_cropped_image_list[1] 





y_vals =np.hstack((np.zeros(len(background_images_subset)),np.ones(len(marked_cropped_image_list[1])))) 

zipped_data =zip(training_data,y_vals)

random.shuffle(zipped_data)

training_data_shuffled,y_vals_shuffled = zip(*zipped_data)

x = training_data_shuffled
y = y_vals_shuffled
#

x =np.array(x)


x = x.reshape(x.shape[0], 1, x[0].shape[0], x[0].shape[1])
y = np.array(y)

training_data_shuffled = x
y_vals_shuffled = y 

x_train_file = 'x_train' 
y_train_file = 'y_train' 

np.save(x_train_file,training_data_shuffled)
np.save(y_train_file,y_vals_shuffled)

#
#f, ax = plt.subplots(2, sharex = True )
#plt.figure
#ax[0].imshow(labeled_image_list[1])
#ax[1].imshow(raw_images_list[1])
#




















#trial rotation 


#
#
#flipped_marked_cropped_list_lr = []
#flipped_marked_cropped_list_ud = [] 
#flipped_marked_cropped_list_90 = []
#for i in marked_cropped_image_list[1]: 
#    flipped_marked_cropped_list_lr.append(np.fliplr(i))
#    flipped_marked_cropped_list_ud.append(np.flipud(i))
#    flipped_marked_cropped_list_90.append(np.rot90(i))
#
#
#    
#marked_cropped_image_list[1] =marked_cropped_image_list[1] +flipped_marked_cropped_list_90 +flipped_marked_cropped_list_lr +flipped_marked_cropped_list_ud
#



#   
#   sample_regions = random.sample(marked_cropped_image_list[0],36)
#   f,axx = plt.subplots(window_size,window_size)
#   plindx = 0 
#   for pl in range(0,window_size): 
#       for pl2 in range(0,window_size):     
#           axx[pl,pl2].imshow(sample_regions[plindx])
#           axx[pl,pl2].scatter(window_size,window_size,color='black')
#           plindx+=1    
  #############################################################################################
# #############################################################################################
#   if not background_pixel_val: 
#             background_pixel_val
#   raw_image =single_raw_image    
#   red_pixel_class = single_labeled_image        
#   list_of_classes =  np.unique(red_pixel_class)        
#   image_list= [[] for x in range(0,len(list_of_classes))]
#        
#        # the number of empty columns are 
#        
#   window =window 
#        
#   plt.imshow(raw_image)
#   f,ax = plt.subplots(2, sharey =True)  
#   list_indx= 0 
#   for j in range(0,len(list_of_classes)): 
#       i = list_of_classes[j] 
#       class_pixel_list = np.where(red_pixel_class == i )
#       grt_indx_x =  np.where(class_pixel_list[0]>window  )  
#       grt_indx_y  =  np.where(class_pixel_list[1]>window )
#       
#       grt_indx_x2 = np.where( class_pixel_list[0]< raw_image.shape[0] -window )
#       grt_indx_y2  =  np.where( class_pixel_list[0]< raw_image.shape[1] -window)
#       
#       grt_indx_x = np.intersect1d(grt_indx_x,grt_indx_x2)
#       grt_indx_y = np.intersect1d(grt_indx_y,grt_indx_y2)
#       
#       grt_indx = np.intersect1d(grt_indx_x, grt_indx_y)
#      
#       class_pixel_list_array = np.asanyarray(class_pixel_list).T
#       
#       class_pixel_list2 = class_pixel_list_array[grt_indx,:]
#       
#       class_pixel_list = [class_pixel_list2[:,0],class_pixel_list2[:,1]]
#       
#       class_pixel_list_array = np.asanyarray(class_pixel_list).T
#       
#       final_pixel_list_array  = class_pixel_list_array
#
#       # this section is if you want to exclude majority of the pixels from the 
#       #back ground. 
#       if (discard_back ==1 & (i == background_pixel_val)): 
#           [grt_zero_x, grt_zero_y ] = np.where(raw_image>background_pixel_val)
#           grt_zero = np.array([grt_zero_x,grt_zero_y]).T 
#           
#           
#           grt_zero_ind = gi.sub2ind(raw_image.shape, grt_zero[:,0],grt_zero[:,1]) 
#           class_pixel_list_ind = gi.sub2ind(raw_image.shape, class_pixel_list_array[:,0],class_pixel_list_array[:,1]) 
#           
#           grt_zero_pixlist_intersect = np.intersect1d(grt_zero_ind,class_pixel_list_ind)
#           
#           grt_zero_list = gi.ind2sub(raw_image.shape,grt_zero_pixlist_intersect)
#           grt_zero_array= np.array([grt_zero_list[0],grt_zero_list[1]]).T
#           final_pixel_list_array = grt_zero_array
#       
#        
#         
#  
#       for pixel in range(0,len(final_pixel_list_array)):
#     
#         pixelx = final_pixel_list_array[pixel,0]
#         pixely = final_pixel_list_array[pixel,1]
#
#          
#         cropped_image = raw_image[pixelx-window:pixelx+window,pixely-window:pixely+window] 
#                
#         image_list[list_indx].append(cropped_image)     
#    
#        
#    
#        
#       ax[j].imshow(single_raw_image)
#       ax[j].scatter(final_pixel_list_array[:,1], final_pixel_list_array[:,0], color='red')
#       
#       list_indx +=1 
# 
# 
# 
# 
# 
 
 
 
 
 
 
 
#   plt.imshow(single_raw_image)
#   plt.scatter(final_pixel_list[1][:,1], final_pixel_list[1][:,0], color='red')
#       
# 
# 
 #############################################################################################
 #############################################################################################
 #############################################################################################     