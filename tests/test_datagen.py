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
from functions import squares_to_tiles, tiles_to_square, sample_squares, load_image, raw_to_labels, plot_row
from process_patches import fetch_batch
import matplotlib.pyplot as plt
from skimage.io import imread
import timeit

class TestDatagen(unittest.TestCase):
    """ Numpy slice test """
    def setUp(self):
        print self._testMethodName
        
    def test_datagen_visualize(self):
        x_aug, y_aug,x,y = fetch_batch('../images/patches/xs/*', '../images/patches/ys/*', 10)
        
        for i,(x_a, y_a) in enumerate(zip(x, y)):
            plot_row([x_a/255, y_a/255], '../plots/temps/', '%d.png'%i)
    
    def test_datagen(self):
        """ Some sanity checks on the datagen method """
        x_aug, y_aug,_,_ = fetch_batch('../images/patches/xs/*', '../images/patches/ys/*', 3)
        
        print((np.unique(x_aug)))
        self.assertGreater(len(np.unique(x_aug)), 20, 'x does not have enough diversity. Something wrong with input.')
        
    def test_onehot(self):
        """ Check that all the y vectors are onhot i.e only have bit set in one place """
        x_aug, y_aug, _, _ = fetch_batch('../images/patches/xs/*', '../images/patches/ys/*', 64)
        
        y_squashed = y_aug[0].sum(axis=-1)
        self.assertEquals(np.unique(y_squashed)[0], 1, 'y data is not onehot')
        
        #s
        
    def test_datagen_speed(self):
        """ Speed test for the datagen method. Datagen should not be slow otherwise it makes training slower """
        execution_time = timeit.timeit("fetch_batch('../images/patches/xs/*', '../images/patches/ys/*', 64)", setup = "from process_patches import fetch_batch", number=1)
        
        self.assertLess(execution_time, 0.6)
        
if __name__ == "__main__":
    unittest.main()
