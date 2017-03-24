# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from functions import resize_crop_image
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import numpy as np

img = imread('woof.png')
img_new = img.copy()
labels2 = np.zeros((img_new.shape[0], img_new.shape[1]))

def labels2img(labels):
    img = np.zeros((labels.shape))
    for channel in xrange(3):
        img[labels[:,:,channel]==1,channel]=1
    img[:,:,3]=1
    img[labels[:,:,3]==1]=[0,0,0,1]
    #plt.imshow(img)
    return img

colors = [[255,0,0,255], [0,255,0,255], [0,0,255,255]]

for channel in [2,1]:
    channel_i = (img[:,:,channel]>200).astype(float)
    channel_resized = resize_crop_image(channel_i, 0.25, 15)
    channel_resized[channel_resized>0]=1
    channel_resized=channel_resized[1:,1:]

    img_new[channel_resized==1]=colors[channel]
    labels2[channel_resized==1]=channel

# Pick out and resize green channel

# Pick out and resize red channel
#red = (img[:,:,0]>200).astype(float)
#red_r = resize_crop_image(red, 0.25, 15)
#red_r[red_r>0]=1
#red_r=red_r[1:,1:]
# 
#plt.imshow(red_r, cmap=plt.cm.gray)

#img_new[green_r==1][:,:,1]=255
#(img_new[red_r==1])[:,:,0]=255
#imsave('')
#imsave('images/colors.png', img_new)


labels = img_new/255
blacks_pos = (labels[:,:,0]==0)&(labels[:,:,1]==0)&(labels[:,:,2]==0)
# All not clearly classified pixels must be black
channels_pos = (labels[:,:,0]>0)|(labels[:,:,1]>0)|(labels[:,:,2]>0)
labels[~channels_pos]=[0,0,0,1]
labels[~blacks_pos,3]=0

labels2[~channels_pos]=3

for i in xrange(4):
    print("Class: %d: %d"%(i, labels[:,:,i].sum()))

# Save labels
np.save('labels.npy', labels)
np.save('labels2.npy', labels2)

plt.imshow(labels2img(labels))
plt.show()
