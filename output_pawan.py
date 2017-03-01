
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
raw_image = cv2.imread('raw_image_cropped.png',0)
colors_heart = raw_image
window_size =3
image_list,final_pix_list = gi.get_images(raw_image,window_size)
output_list = []
#image_list =np.array(image_list)
#image_list2  = image_list.reshape(image_list.shape[0], 1, image_list[0].shape[0], image_list[0].shape[1])
count = 0
shape_list = []
image_list = np.array(image_list)
im = image_list.reshape(image_list.shape[0],1,image_list[0].shape[0], image_list[0].shape[1])

import time
start_time = time.time()
print ("Starting now!!!")
output = model.predict(im)
print("--- %s seconds ---" % (time.time() - start_time))

for i in range(0,len(output)): 
     
#     image_list2.append(i.reshape(1,window_size))
#     print(i)
     # output = model.predict(i.reshape(1,1,window_size*2,window_size*2))

#     output_list.append(output)
     
     pixelx = final_pix_list[count,0]
     pixely = final_pix_list[count,1]
     if output[i]>0.5:
         
            colors_heart[pixelx,pixely] =255
           

     count += 1 
     # print(count)

plt.imshow(colors_heart)
plt.show()
sys.exit()