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
from functions import squares_to_tiles, tiles_to_square

class TestConvSquare(unittest.TestCase):
    """ Numpy slice test """
    def setUp(self):
        print self._testMethodName

    def test_squares_to_tiles(self):
        """ Does generating squares tiles work? Tests both conv_stride and conv_squares """
        data = np.ones((8,8))
        squares = squares_to_tiles(data, (2,2), (2,2))
        #print(len(squares), squares[0])
        
        self.assertEqual(len(squares), 16, 'We did not get 16 squares')
        [self.assertEqual(sq.shape, (2,2), 'Square shape is incorrect') for sq in squares]
        
    def test_tiles_to_square(self):
        """ Can you generate a larger 3,3 square by averaging 4 smaller 2,2 squares with stride 1,1 """
        squares = [np.ones((2,2))]*4
        
        larger_square = tiles_to_square(squares, (3,3), (2,2), (1,1))
        self.assertEqual(larger_square.shape, (3,3), 'Tiles to square failing!')
        
    def test_square_tile_square(self):
        """ Test square -> tiles -> squares """
        data = np.ones((8,8))
        square_shape = (2,2)
        squares = squares_to_tiles(data, (2,2), (2,2))
        data2 = tiles_to_square(squares, data.shape, square_shape, (2,2))
        np.testing.assert_equal(data, data2, 'Identity is not preserved')
        #self.assertEqual(data, data2, 'Identity is not preserved')
        
if __name__ == "__main__":
    unittest.main()
    
#squares = [np.ones((2,2))]*4
#
#indices = stride_indices(arr_shape, sq_size, stride)
#    
#arr = np.zeros(arr_shape)
#weight_square = np.ones(tuple(sq_size))
#
#weights = np.zeros(arr_shape)
#
#for i, inds in enumerate(indices):
#    arr[inds] += squares[i]
#    weights[inds] += weight_square
#    
#return arr/weights
#data = np.ones((3,3))
#sq_size = (2,2)
#stride = (1,1)
#arr_shape=data.shape
#get_tile_indices = lambda i: xrange(0, arr_shape[i]-sq_size[i]+stride[i], stride[i])
#prods = itertools.product(*map(get_tile_indices, xrange(len(sq_size))))
#
#prods2 = list(prods)
#print(prods2)
#
#slicer = lambda x: slice(*x)
#list_slices = lambda prod: map(slicer, zip(prod, prod+sq_size))
#square_indices = map(list_slices, prods2)
#print(square_indices)