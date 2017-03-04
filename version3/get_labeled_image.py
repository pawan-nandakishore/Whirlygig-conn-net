# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 19:26:15 2017

@author: pawan
"""

import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import Get_images_for_classes_functions as gi 
from skimage.transform import rotate
from PIL import Image
import random

plt.close('all')
  
  

      
marked_image = cv2.imread('marked_image')

marked_image2 = cv2.imread('marked_image2')
raw_image = cv2.imread('raw_image',0)
raw_image2 = cv2.imread('raw_image2',0)
raw_image3 = cv2.imread('raw_image3',0)


scale  = 0.25
cut_off_percent =15.0 
num_of_transformations = 1
#angle_list = [0.0,25.0,45.0,75.0,90.0,135.0,155.0,180.0,195.0,225.0,270.0, 285.0,320.0]
#angle_list = [25.0, 45.0, 90.0,135.0,180.0,215.0]
angle_list = [random.uniform(0,360) for x in range(0,6)] 



#
#raw_image2 = gi.resize_crop_image(raw_image2,scale,cut_off_percent)
#
#marked_image2 = gi.resize_crop_image(marked_image2,scale,cut_off_percent)
#unprocessed_marked_images = [marked_image, marked_image2]
#unprocessed_raw_images = [raw_image,raw_image2]


unprocessed_marked_images = [marked_image, marked_image2]
unprocessed_raw_images = [raw_image, raw_image2]

list_of_marked_images = gi.get_image_transformations(unprocessed_marked_images,angle_list,scale,cut_off_percent)







list_of_raw_images = gi.get_image_transformations(unprocessed_raw_images,angle_list,scale,cut_off_percent)


#
#list_of_raw_images =  gi.generate_raw_image_list(list_of_marked_images,list_of_raw_images,num_of_transformations)
#


#
    
#img = list_of_raw_images[0]
#img2 = list_of_images[1]
#img3 = list_of_images[2]

#
#
#f, ax = plt.subplots(2, sharey = True )
#plt.figure
#ax[0].imshow(marked_image2)
#ax[1].imshow(raw_image2)
#ax[2].imshow(img3)
#
#



masks = [ np.zeros(list_of_marked_images[0].shape[0:2]) for i in range(0,len(list_of_marked_images))  ] 

#f,ax =plt.subplots(len(list_of_marked_images))
for i in range(0,len(masks)): 
   single_marked_image = list_of_marked_images[i] 
   single_mask = masks[i]
   marked_pixels  =  np.where(single_marked_image [:,:,0] !=single_marked_image [:,:,2])
   single_mask[marked_pixels[0], marked_pixels[1]] = 255
#   ax[i].imshow(single_marked_image)
#   ax[i].scatter(marked_pixels[1],marked_pixels[0], color='red')
#   
#   
   masks[i] =single_mask
   















#
#marked_pixels= np.where(marked_image[:,:,0] != marked_image[:,:,2])
#mask[marked_pixels[0],marked_pixels[1] ] = 255
#
#marked_pixels = np.array([marked_pixels[0],marked_pixels[1]])


#
#plt.figure
#plt.imshow(raw_image)

#
# plt.imshow(single_marked_image)
#   plt.scatter(marked_pixels[1],marked_pixels[0], color='red')
filename= 'labeled_images' 
#
np.save(filename,masks)

filename= 'raw_images_cropped' 
np.save(filename, list_of_raw_images)

#
#filename= 'raw_image_cropped2' 
#plt.imsave(filename, raw_image2)
#
#
#
#
#
#








#marked_image = cv2.resize(marked_image,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
#raw_image = cv2.resize(raw_image,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
#
#marked_image2 = marked_image.copy()
#
#cut_off_vals = [marked_image.shape[0]*(cut_off_percent/100), marked_image.shape[1]*(cut_off_percent/100)] 
#
#end_vals = [marked_image.shape[0]-cut_off_vals[0],marked_image.shape[1]-cut_off_vals[1]] 
#
#end_vals= np.round(end_vals) 
#marked_image =marked_image[cut_off_vals[0]:end_vals[0],cut_off_vals[1]:end_vals[1]  ]
#raw_image =raw_image[cut_off_vals[0]:end_vals[0],cut_off_vals[1]:end_vals[1]  ]
#


#
#markingLower = np.array([0 ,  0 , 230 ],dtype=np.uint8)
#markingUpper = np.array([0,  0 , 255],dtype=np.uint8)



#def resize_crop_image(image,scale,cutoff_percent):
#    image = cv2.resize(image,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
#    cut_off_vals = [image.shape[0]*(cut_off_percent/100), image.shape[1]*(cut_off_percent/100)]     
#    cut_off_vals = np.round(cut_off_vals)
#    end_vals = [image.shape[0]-cut_off_vals[0],image.shape[1]-cut_off_vals[1]] 
#    end_vals= np.round(end_vals) 
#    image =image[cut_off_vals[0]:end_vals[0],cut_off_vals[1]:end_vals[1]  ]
#    return(image)
#
#
#
#def image_rotation(image,angle_list): 
#  list_of_images = []
#  img = Image.fromarray(image)
#   
#  for i in angle_list: 
#      img2 = img.rotate(i)
#      img2 =np.array(img2)      
#      img2 = img2[0:raw_image2.shape[0], 0:raw_image2.shape[1]]
#      list_of_images.append(img2)
#      
#  return(list_of_images)    
#
#
#def get_image_transformations(image_list,angle_list,scale,cutoff_precent,raw_check): 
#  rotated_resized_images = []
#  flipped_lr_images = []
#  flipped_ud_images = []
#  for image in image_list:    
#    resized_images = resize_crop_image(image,scale,cutoff_precent)    
#    if(raw_check==0):     
#        rotated_resized_images += image_rotation(resized_images, angle_list )  
#    else: 
#          rotated_resized_images=   [resized_images]
#  
#  if(raw_check== 0) :    
#      for image in rotated_resized_images:    
#          flipped_lr_images.append(np.fliplr(image))
#          flipped_ud_images.append(np.flipud(image))
#   
#      rotated_resized_images  += flipped_lr_images +flipped_ud_images 

#def generate_raw_image_list(list_of_marked_images,list_of_raw_images) :
#
#    new_list_of_raw_images = []
#    
#    num_of_transformations = 9 
#    
#    if(len(list_of_marked_images)%num_of_transformations == 0): 
#        num_of_raw_images = len(list_of_marked_images)/num_of_transformations
#    
#        for i in range(0,num_of_raw_images): 
#           for reps in range(0,len(list_of_marked_images)):
#                new_list_of_raw_images.append( list_of_raw_images[i])
#    else: 
#        print('error, issue with number of transformations' )
#
#    return(new_list_of_raw_images)
#    