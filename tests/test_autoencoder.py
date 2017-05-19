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
from models import pawannet, keras_mnist_autoencoder, keras_mnist_autoencoder_loaded
from keras.callbacks import LambdaCallback
from keras.datasets import mnist
from keras.models import load_model

#def autoencoder_yield_batch(x, y, n=64, patch_size=56, preprocess=False, augment=False, crop_size=20):  
#    """ Yields batch of size n infinitely """
#    
#    while True:
#        x_aug, y_aug = fetch_batch(x, y, n, patch_size, preprocess, augment, crop_size)
#        x_aug = x_aug/255
#        y_aug = y_aug/255
#        yield (x_aug, y_aug)
#
#def save_mod(epoch, logs):
#    global count
#    global autoencoder
#    if count%40==0:
#        print('Saving model, count: %d'%count)
#        model.save('../models/%d.h5'%count)
#    count+=1
#
#count = 0
#cb = LambdaCallback(on_batch_begin=save_mod)
#
#model = pawannet((56,56,3), (36,36,3), 10, kernel=3)
#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#x, y = read_data(glob.glob('../images/cropped/rgbs/*'), glob.glob('../images/cropped/rgbs/*'))
#dataGenerator = autoencoder_yield_batch(x, y, n=64, patch_size=56, preprocess=False, augment=False, crop_size=20)
#model.fit_generator(dataGenerator, samples_per_epoch = 600, nb_epoch = 30, callbacks=[cb])

def yield_batch(x, y):  
    """ Yields batch of size n infinitely """
    
    while True:
        idx = np.random.choice(np.arange(len(x)), 128, replace=False)
        x_sample = x[idx]
        y_sample = y[idx]
        yield (x_sample, y_sample)

class TestAutoencoder(unittest.TestCase):
    """ Numpy slice test """
    def setUp(self):
        print self._testMethodName
        
        
    def load_mnist_data(self):
        """ Helper to preprocess and load mnist data """
        (x_train, _), (x_test, _) = mnist.load_data()
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        
        # Convert to floatdd
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        
        return x_train, x_test
    
    def run_experiment(self, model):
        """ Common helper for both """
        x_train, x_test = self.load_mnist_data()
        dataGenerator = yield_batch(x_train, x_train)
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        #model.fit_generator(dataGenerator, samples_per_epoch = 600, nb_epoch = 30, callbacks=[])
        model.fit(x_train, x_train, epochs=3, batch_size=128, shuffle=True, validation_data=(x_test, x_test))
        model.save('mnist_auto.h5')
    
    def _test_autoencoder(self):
        """ Test that pawannet is able to reconstruct the patches from input image. This is always the task to be done first to check that everything first. """
        pass
        
    def _test_pawannet_mnist(self):
        """ Check that pawannet is able to learn fast on the mnist dataset to a reasonable accuracy """
        self.run_experiment(pawannet((28,28,1), (28,28,1), 0, kernel=3))
        
    def test_kerasnet_mnist(self):
        """ Test that keras convnet is able to learn fast on the mnist dataset to a reasonable accuracy """
        self.run_experiment(keras_mnist_autoencoder((28,28,1)))
    
    def test_zautoencoder_mnist_visualize(self):
        """ Generate output on mnist """
        x_train, x_test = self.load_mnist_data()
        model = load_model('mnist_auto.h5')
        
        x_out = model.predict(x_test)
        x_out = x_out.reshape((x_test.shape[0], 28,28))
        
        for i,img in enumerate(x_out[0:10]):
            imsave('outs/%d.png'%i, img)
        # Generate batch of images 10
        
        # Generate output
        
        # Run loop and save output on all of these
        
        #self.assertEqual(data, data2, 'Identity is not preserved')
        
if __name__ == "__main__":
    unittest.main()
