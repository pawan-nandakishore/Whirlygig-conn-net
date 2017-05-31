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
from unittest2 import TestCase
from functions import squares_to_tiles, tiles_to_square, load_image, raw_to_labels
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.draw import ellipse, ellipse_perimeter
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.morphology import distance_transform_edt
from skimage.measure import label, regionprops
from process_patches import fetch_batch, read_data, tensor_blur
import glob
from models import pawannet_autoencoder, keras_mnist_autoencoder
from keras.callbacks import LambdaCallback
from keras.datasets import mnist
from keras.models import load_model
from scipy.stats import entropy
from keras.callbacks import History 

class TestPawannetWhirlygig(TestCase):
    """ Testing pawannet autoencoder on whirlygig images """
    
    def setUp(self):
        self.count = 0
        self.model = pawannet_autoencoder((56,56,3), (36,36,3), 10, kernel=3)
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        self.cb = LambdaCallback(on_batch_begin=self.save_mod)
    
    def yield_batch(self, x, y, n=64, patch_size=56, preprocess=False, augment=False, crop_size=20):  
        """ Yields batch of size n infinitely """
        
        while True:
            x_aug, y_aug = fetch_batch(x, y, n, patch_size, preprocess, augment, crop_size)
            
            x_aug = x_aug/255
            y_aug = y_aug/255
            yield (x_aug, y_aug)
            
    def save_mod(self, epoch, logs):
        if self.count%40==0:
            print('Saving model, count: %d'%self.count)
            self.model.save('../models/%d.h5'%self.count)
        self.count+=1
        
    def test_whirlygig_loss(self):
        """ Train network on whirlygig images and check that the training loss converges """
        history = History()
        x, y = read_data(glob.glob('../images/cropped/rgbs/*'), glob.glob('../images/cropped/rgbs/*'))
        
        # Testing by visualizing
        x_test,y_test = fetch_batch(x, y, n=10, patch_size=56, preprocess=False, augment=True, crop_size=20)
        [imsave('%d_i.png'%i, img.astype(float)/255) for i,img in enumerate(x_test)]
        [imsave('%d_o.png'%i, img.astype(float)/255) for i,img in enumerate(y_test)]
        
        # Run the network
        dataGenerator = self.yield_batch(x, y, n=64, patch_size=56, preprocess=False, augment=True, crop_size=20)
        self.model.fit_generator(dataGenerator, samples_per_epoch = 600, nb_epoch = 1, callbacks=[self.cb, history])
        
        self.assertGreater(history.history['val_acc'][0], 0.4)

class TestAutoencoder(TestCase):
    """ Testing pawannet autoencoder on mnist images """
    
    def yield_batch(x, y):  
        """ Yields batch of size n infinitely """
    
        while True:
            idx = np.random.choice(np.arange(len(x)), 128, replace=False)
            x_sample = x[idx]
            y_sample = y[idx]
            yield (x_sample, y_sample)
    
    def setUp(self):
        print self._testMethodName
        
    def load_mnist_data(self):
        """ Helper to preprocess and load mnist data """
        (x_train, _), (x_test, _) = mnist.load_data()
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        
        # Sample 10,000 examples from x_train
        idx = np.random.choice(np.arange(x_train.shape[0]), 10000, replace=False)
        x_train = x_train[idx]
        
        # Convert to floatdd
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        
        return x_train, x_test
    
    def run_experiment(self, model, x_train, y_train, x_test, y_test):
        """ Common helper for both """
        #dataGenerator = yield_batch(x_train, y_train)
        history = History()
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        #model.fit_generator(dataGenerator, samples_per_epoch = 600, nb_epoch = 30, callbacks=[])
        model.fit(x_train, x_train, nb_epoch=1, batch_size=64, shuffle=True, validation_data=(x_test, y_test), callbacks=[history])
        model.save('mnist_auto.h5')
        return model, history
    
    def _test_autoencoder(self):
        """ Test that pawannet is able to reconstruct the patches from input image. This is always the task to be done first to check that everything first. """
        pass
        
    def _test_pawannet_mnist(self):
        """ Check that pawannet is able to learn fast on the mnist dataset to a reasonable accuracy """
        x_train, x_test = self.load_mnist_data()
        modelInit = pawannet_autoencoder((28,28,1), (28,28,1), 0, kernel=3)
        model, hist = self.run_experiment(modelInit, x_train, x_train, x_test, x_test)
        
        # Test that accuracy greater than 60% and loss<0.3
        self.assertLess(hist.history['val_loss'][0], 0.3)
        self.assertGreater(hist.history['val_acc'][0], 0.6)
        
#    def _test_kerasnet_mnist(self):
#        """ Test that keras convnet is able to learn fast on the mnist dataset to a reasonable accuracy """
#        x_train, x_test = self.load_mnist_data()
#        modelInit = keras_mnist_autoencoder((28,28,1))
#        model, history = self.run_experiment(modelInit, x_train, x_train, x_test, x_test)
        
        # Check that it's similar to original
        #x_out = model.predict(x_test)
        #np.testing.assert
        
    def _visualize(self):
        """ Visualize the outputs of a model """
        x_out = model.predict(x_test)
        x_out = x_out.reshape((x_test.shape[0], 28,28))
        
        x_out = model.predict(x_test)
        x_out = x_out.reshape((x_test.shape[0], 28,28))
        
        for i,img in enumerate(x_out[0:10]):
            imsave('outs/%d.png'%i, img)
        
if __name__ == "__main__":
    unittest.main()
