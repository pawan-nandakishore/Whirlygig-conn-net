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

labeled_image_list = np.load('labeled_images.npy')

raw_images_list = np.load('raw_images_cropped.npy')

labeled_image_list = list(labeled_image_list)
raw_images_list = list(raw_images_list)



number_of_classes = 2
window_size = 3
discard_back =1
background_pixel_val = 0 
window= window_size
#marked_cropped_image_list = gi.get_images_per_class(labeled_image,raw_image,0, 1,window_size)
all_marked_cropped_image_list =[[] for i in range(0,number_of_classes)]


for i in range(0,len(labeled_image_list)): 
   single_labeled_image =  labeled_image_list[i]
   single_raw_image = raw_images_list[i]
 
   marked_cropped_image_list =  gi.get_images_per_class(single_labeled_image,single_raw_image,0, 1,window_size)
                
   all_marked_cropped_image_list[0] += marked_cropped_image_list[0]
   all_marked_cropped_image_list[1] += marked_cropped_image_list[1]
   


marked_cropped_image_list =all_marked_cropped_image_list

percent_of_bck = 50
percent_of_bck = percent_of_bck/100 

background_images= marked_cropped_image_list[0]
background_images_subset = random.sample(background_images,len(marked_cropped_image_list[1])+np.round(len(marked_cropped_image_list[0])*percent_of_bck) )



training_data = background_images_subset +marked_cropped_image_list[1] 


y_vals =np.hstack((np.zeros(len(background_images_subset)),np.ones(len(marked_cropped_image_list[1])))) 

zipped_data =zip(training_data,y_vals)

random.shuffle(zipped_data)

training_data_shuffled,y_vals_shuffled = zip(*zipped_data)

x = training_data_shuffled
y = y_vals_shuffled

x = np.array(x)
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
