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
from functions import squares_to_tiles, tiles_to_square, sample_squares, load_image, raw_to_labels
import matplotlib.pyplot as plt
from skimage.io import imread

class TestCrops(unittest.TestCase):
    """ Numpy slice test """
    
    def _test_sample_square_xy_correspondence(self):
        """ Test that the difference between x and y is less than equal to certain threshold """
        print('lulz')
        #self.assertEqual(data, data2, 'Identity is not preserved')
        
    def _test_sample_square_x_normalized(self):
        """ Test that all inputs are normalized """
        print('Keep it large')
        
    def _test_sample_square_y_softmax(self):
        """ Test that only one of y is 1 in the whole stack """
        print('haha')
        
    
#    def _calculate_ratio(self, squares):
#        """ Calculates the ratio of squares of each type majorly r, majorly b, majorly g in squares """
#        zz=squares.sum(axis=(1,2))
#        groo = np.argmax(zz, axis=1)
#        #counts = np.bincount(groo)
#        
#        counts = np.array([squares[...,].sum() for c in xrange(4)])
#        
#        ratios_calculated = ((counts+1e-5)/counts.sum())
#        
#        
#        return ratios_calculated
#    
#    def _initialize_img(self):
#        shape = 100
#        img = np.zeros((2,shape,shape,4))
#        ratios = [0.3,0.2,0.1,0.4]
#        prev = 0
#    
#        for c,r in enumerate(ratios):
#            r_next = int(prev+r*shape)
#            print(r_next)
#            img[:,prev:r_next,:,c] = 1
#            prev = r_next
#            
#        print(img.shape)
#           
#        return img, ratios
#    
#    def _load_img(self):
#        img = raw_to_labels(imread('../images/labeled/1.png'))
#        img_tensor = np.array([img, img])
#        
#        ratios = [0.25,0.25,0.25,0.25]
#        
#        return img_tensor, ratios
#        #plt.imshow(img)
#        
#    def _test_foveal_sampling_hyperstack(self):
#        """ Given a black and a white image, make sure that no sampling from the white image is done """
#        
#    def _test_foveal_sampler(self):
#        """ Test that in a rgb image with 70% red, 20% green, 10% blue, most of the squares are primarily red, then green, then blue """
#        #img, ratios = self.initialize_img()
#        img, ratios = self.load_img()
#        #plt.imshow(img1[0])
#        #print('koko')
#        #self.assertEqual(img1.shape, img2.shape)
#        #print(ratios)
#        
#        squares = sample_squares(img, 10000, np.ones((1,36,36,4)))
#        #import pdb; pdb.set_trace()
#        print(ratios, self.calculate_ratio(squares))
#        #print(squares.shape)
#        
#        #np.testing.assert_almost_equal(ratios, self.calculate_ratio(squares), 1)
        
if __name__ == "__main__":
    unittest.main()
    
#""" Linear convolution sampling """
#
#
#
#""" Find median weights for each pixel """
#img = np.random.rand(4,10,10,3)
#kernel = np.ones((1,3,3,3))
#img_weighted = convolve(img, kernel, mode='valid')
#
#
#img_weighted = img_weighted.reshape(4,8,8)
#print(img_weighted.shape)
#
#""" Now just find the correct kernel and weights """
#
#print(img_weighted.shape)
#img_flat = img_weighted.flatten()
#img_flat /= img_flat.sum()
#img_flat_indices = np.arange(img_flat.shape[0])
#print(img_flat_indices)
#choices = np.random.choice(img_flat_indices, 3, p=img_flat)
#print(choices, choices.shape)
#peta=np.unravel_index(choices[0], img.shape)
#print(peta)
#
#""" Show that in a rgb image, it doesn't pick much of 
##indices = np.indices(img.shape)
#
##print(rez, indices)"""