# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 10:19:42 2017

@author: pawan
"""
import numpy as np
from scipy.io import loadmat
import random
import matplotlib.pyplot as plt
import os
import cv2 
plt.rcParams['axes.facecolor'] = 'black'
boxLower = np.array([0, 0,0.9 ], dtype =np.float)
boxUpper = np.array([0.1,0.1,1.0 ], dtype =np.float)


boxLower2 = np.array([0, 0.2, 0 ], dtype =np.float)
boxUpper2 = np.array([0.2,1,0.2 ], dtype =np.float)






plt.close('all')

boundry_rgb =  cv2.imread('boundries3.png')

boundry_rgb2 = plt.imread('boundaries3.png')
boundry_rgb22 =  cv2.imread('/media/pawan/0B6F079E0B6F079E/PYTHON_SCRIPTS/Data science challenges/whirlygig/boundries3.png',0)

plt.figure()
plt.imshow(boundry_rgb2)


colour_frame = cv2.cvtColor(boundry_rgb2, cv2.COLOR_BGR2RGB)
#gray = cv2.cvtColor(boundry_rgb22,cv2.COLOR_BGR2GRAY)

gray =boundry_rgb22
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

red_mask = cv2.inRange(colour_frame,boxLower,boxUpper)
green_mask = cv2.inRange(colour_frame,boxLower2,boxUpper2)



plt.imshow(green_mask)


plt.figure()
plt.imshow(colour_frame)


red_pixels = np.where(red_mask==255)
green_pixels = np.where(green_mask==255)

image2 = np.zeros(red_mask.shape)

image2[red_pixels[0], red_pixels[1]] = 255
image2[green_pixels[0], green_pixels[1]] = 255


plt.figure()
plt.imshow(image2)