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
from functions import sample_patches, squares_to_tiles, tiles_to_square, load_image, raw_to_labels, plot_row
from process_patches import read_data, augment_tensor, crop_tensor, fetch_batch
import matplotlib.pyplot as plt
from skimage.io import imread
import timeit
import glob

class TestDatagen(unittest.TestCase):
    """ Numpy slice test """
    def setUp(self):
        print self._testMethodName
        
    def test_datagen_visualize(self):
        x, y = read_data(glob.glob('../images/cropped/rgbs/*'), glob.glob('../images/cropped/labeled/*'))
        x_patches, y_patches = fetch_batch(x, y, 64, 56, True, True, 20)
        
        for i,(x_a, y_a) in enumerate(zip(x_patches, y_patches)):
            plot_row([x_a, np.argmax(y_a, axis=-1)], '../plots/temps/', '%d.png'%i)
    
    def test_datagen(self):
        """ Some sanity checks on the datagen method """
        x, y = read_data(glob.glob('../images/cropped/rgbs/*'), glob.glob('../images/cropped/labeled/*'))
        x_patches, y_patches = fetch_batch(x, y, 64, 56, True, True, 20)
        
        print((np.unique(x_patches)))
        self.assertGreater(len(np.unique(x_patches)), 20, 'x does not have enough diversity. Something wrong with input.')
        
    def test_onehot(self):
        """ Check that all the y vectors are onhot i.e only have bit set in one place """
        x, y = read_data(glob.glob('../images/cropped/rgbs/*'), glob.glob('../images/cropped/labeled/*'))
        x_patches, y_patches = fetch_batch(x, y, 64, 56, True, True, 20)
        
        y_squashed = y_patches[0].sum(axis=-1)
        self.assertEquals(np.unique(y_squashed)[0], 1, 'y data is not onehot')
        
    def _test_datagen_speed(self):
        """ Speed test for the datagen method. Datagen should not be slow otherwise it makes training slower """
        x, y = read_data(glob.glob('../images/cropped/rgbs/*'), glob.glob('../images/cropped/labeled/*'))
        execution_time = timeit.timeit("fetch_batch(x, y, 64, 56, True, 20)", setup = "from process_patches import fetch_batch", number=1)
        
        self.assertLess(execution_time, 0.6)
        
    def test_sample_patches(self):
        """ Test that there are no errors in sampling squares """
        x, y = read_data(glob.glob('../images/cropped/rgbs/*'), glob.glob('../images/cropped/labeled/*'))
        x_patches, y_patches = fetch_batch(x, y, 64, 56, True, True, 20)
        
        self.assertEqual(x_patches.shape, (64,56,56,3), 'Sampled x shape incorrect')
        self.assertEqual(y_patches.shape, (64,36,36,4), 'Sampled y shape incorrect')
        
        
    def test_xy_correspondence_sampling_patches(self):
        """ Test that on sampling patches, each x corresponds to each y """
        x, y = read_data(glob.glob('../images/cropped/labeled/*'), glob.glob('../images/cropped/labeled/*'))
        x_patches, y_patches = fetch_batch(x, y, 64, 56, True, True)
        x_patches = np.array([raw_to_labels(y*255) for y in y_patches])
        
        np.testing.assert_array_equal(x_patches, y_patches, 'XY correspondence not being met')
        
    def test_sampling_uniform_distribution(self):
        """ Test that the sampled squares come from a uniform distribution """
        pass
        
if __name__ == "__main__":
    unittest.main()
