# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 23:42:06 2017

@author: pawan
"""

import numpy as np
from scipy.io import loadmat
import random
import matplotlib.pyplot as plt
import os
import cv2 
import re




boundary_loc = '/media/pawan/0B6F079E0B6F079E/PYTHON_SCRIPTS/Data science challenges/whirlygig/boundaries_5x/'

files_boundary = os.listdir(boundary_loc)
all_mats_boundary = []
pixels_vals = []

y_vals_boundary =[]



for i in range(0,len(files_boundary)):
     mats = np.load( boundary_loc+files_boundary[i])
     mats = mats.flatten().T
     all_mats_boundary.append(mats)
     
     file_name = files_boundary[i]
     underscore_val1 = file_name.split('_')[2]
     underscore_val2 = file_name.split('_')[3]
     underscore_val2 = underscore_val2.split('.')[0]
     pix_val = [int(underscore_val1),int(underscore_val2)]
     pixels_vals.append(pix_val)
     
     
     
     
[ y_vals_boundary.append([1,0,0]) for x in range(len(files_boundary))]


all_mats_boundary = np.asanyarray(all_mats_boundary)






exteriors_loc = '/media/pawan/0B6F079E0B6F079E/PYTHON_SCRIPTS/Data science challenges/whirlygig/exteriors_5x/'

files_exteriors = os.listdir(exteriors_loc)
all_mats_exteriors = []
pixels_vals_exteriors = []
y_vals_exteriors =[]



for i in range(0,len(files_exteriors)):
     mats = np.load( exteriors_loc+files_exteriors[i])
     mats = mats.flatten().T
     all_mats_exteriors.append(mats)
     
     file_name = files_exteriors[i]
     underscore_val1 = file_name.split('_')[2]
     underscore_val2 = file_name.split('_')[3]
     underscore_val2 = underscore_val2.split('.')[0]
     pix_val = [int(underscore_val1),int(underscore_val2)]
     pixels_vals_exteriors.append(pix_val)
     
     


[ y_vals_exteriors.append([0,1,0]) for x in range(len(files_exteriors))]
   
#y_vals_exteriors = np.zeros(len(files_exteriors))

all_mats_exteriors = np.asanyarray(all_mats_exteriors)






interiors_loc = '/media/pawan/0B6F079E0B6F079E/PYTHON_SCRIPTS/Data science challenges/whirlygig/interiors_5x/'

files_interiors = os.listdir(interiors_loc)
all_mats_interiors = []
pixels_vals_interiors = []

y_vals_interiors =[]


for i in range(0,len(files_interiors)):
     mats = np.load( interiors_loc+files_interiors[i])
     mats = mats.flatten().T
     all_mats_interiors.append(mats)
     
     file_name = files_interiors[i]
     underscore_val1 = file_name.split('_')[2]
     underscore_val2 = file_name.split('_')[3]
     underscore_val2 = underscore_val2.split('.')[0]
     pix_val = [int(underscore_val1),int(underscore_val2)]
     pixels_vals_interiors.append(pix_val)
     
     
     
[ y_vals_interiors.append([0,0,1]) for x in range(len(files_interiors))]     
#y_vals_interiors = 2*np.ones(len(files_interiors))

all_mats_interiors = np.asanyarray(all_mats_interiors)





length_mat  = all_mats_boundary.shape[0]
length_mat = length_mat/3
all_train_1 = all_mats_boundary[0:length_mat,:]
all_train_2 = all_mats_exteriors[0:length_mat,:]
all_train_3 = all_mats_interiors[0:length_mat,:]

all_y_1 = y_vals_boundary[0:length_mat]
all_y_2 = y_vals_exteriors[0:length_mat]
all_y_3 = y_vals_interiors[0:length_mat]



all_y_1 = np.asanyarray(all_y_1)
all_y_2 = np.asanyarray(all_y_2)
all_y_3= np.asanyarray(all_y_3)
all_y = np.vstack((all_y_1,all_y_2) )
all_y = np.vstack((all_y,all_y_3 ))


#y_vals = [y_vals_boundary,y_vals_exteriors, y_vals_interiors]


all_train = np.vstack((all_train_1,all_train_2))
all_train = np.vstack((all_train,all_train_3)) 

filename = 'y_vals' 
np.save(filename,all_y)


filename = 'training_data'
np.save(filename, all_train)
