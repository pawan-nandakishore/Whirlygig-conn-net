
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 17:39:50 2017

@author: pawan
"""
#
from keras.models import load_model
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from skimage.color import gray2rgb
from skimage.io import imsave
import cv2
import Get_images_for_classes_functions as gi 
import sys 
model = load_model('conv.h5')
filename = 'raw_image_cropped3'
filename2 = filename+'.png'
raw_image = cv2.imread(filename2,0)
labeled_image_list = np.load('labeled_images.npy')

# single_labeled_image = labeled_image_list[1]

print(raw_image.shape)
# print(single_labeled_image.shape)


colors_heart = raw_image
window_size =6
image_list,final_pix_list = gi.get_images(raw_image,window_size)
output_list = []
#image_list =np.array(image_list)
#image_list2  = image_list.reshape(image_list.shape[0], 1, image_list[0].shape[0], image_list[0].shape[1])
count = 0
shape_list = []
image_list = np.array(image_list)
im = image_list.reshape(image_list.shape[0],1,image_list[0].shape[0], image_list[0].shape[1])
im = im/255.0 

import time
start_time = time.time()
print ("Starting now!!!")
output = model.predict(im)
print("--- %s seconds ---" % (time.time() - start_time))
correct_val=0
for i in range(0,len(output)): 
     
#     image_list2.append(i.reshape(1,window_size))
#     print(i)
     # output = model.predict(i.reshape(1,1,window_size*2,window_size*2))

#     output_list.append(output)
     # output[np.where(output>0.5)]= 255
     # output[np.where(output<0.5)]= 0
      
     pixelx = final_pix_list[count,0]
     pixely = final_pix_list[count,1]
     # class_val = single_labeled_image[pixelx,pixely]
     # print(output[1], class_val)
# & (output[i]==class_val)
     if ((output[i]>0.5) ) :
            # print('correct:', output[i],class_val)
            colors_heart[pixelx,pixely] =255
            correct_val +=1
     # elif ((output[i]>0.5) & (output[i]!=class_val)):
     #         # print('wrong:', output[i],class_val)
     #         colors_heart[pixelx,pixely] =127
     count += 1 

     # print(count)
    
           
     #        count += 1 


misclass_ratio = np.float(correct_val)/ np.float(len(output))

print(correct_val,'  ' ,len(output))
plt.imshow(colors_heart)
# filename = filename+'predicted_class' +'window_size'+str(window_size)
# plt.imsave(filename,colors_heart)
plt.show()
# 