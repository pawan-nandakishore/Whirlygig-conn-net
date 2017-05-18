#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 01:46:36 2017

@author: monisha
"""
import sys
sys.path.append('..')

import numpy as np
import unittest
import itertools
from functions import squares_to_tiles, tiles_to_square, load_image, raw_to_labels
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.draw import ellipse, ellipse_perimeter
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.morphology import distance_transform_edt
from skimage.measure import label, regionprops
from process_patches import fetch_batch, read_data
import glob
from models import pawannet
from keras.callbacks import LambdaCallback

def autoencoder_yield_batch(x, y, n=64, patch_size=56, preprocess=False, augment=False, crop_size=20):  
    """ Yields batch of size n infinitely """
    
    while True:
        x_aug, y_aug = fetch_batch(x, y, n, patch_size, preprocess, augment, crop_size)
        x_aug = x_aug/255
        y_aug = y_aug/255
        yield (x_aug, y_aug)

def save_mod(epoch, logs):
    global count
    global autoencoder
    if count%40==0:
        print('Saving model, count: %d'%count)
        model.save('../models/%d.h5'%count)
    count+=1

count = 0
cb = LambdaCallback(on_batch_begin=save_mod)

model = pawannet((56,56,3), (36,36,3), 10, kernel=3)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
x, y = read_data(glob.glob('../images/cropped/rgbs/*'), glob.glob('../images/cropped/rgbs/*'))
dataGenerator = autoencoder_yield_batch(x, y, n=64, patch_size=56, preprocess=False, augment=False, crop_size=20)
model.fit_generator(dataGenerator, samples_per_epoch = 600, nb_epoch = 30, callbacks=[cb])

class TestAutoencoder(unittest.TestCase):
    """ Numpy slice test """
    def setUp(self):
        print self._testMethodName
    
    def test_autoencoder(self):
        """ Test that pawannet is able to reconstruct the patches from input image. This is always the task to be done first to check that everything first. """
        pass
       
        
    def test_autoencoder_mnist(self):
        """ Check that pawannet is able to autoencode on the mnist dataset """
        pass
        
        #self.assertEqual(data, data2, 'Identity is not preserved')
        
if __name__ == "__main__":
    unittest.main()