#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 01:46:36 2017

@author: monisha
"""

import numpy as np
import unittest

class VisualizeTestCase(unittest.TestCase):
    """ Numpy slice test """

    def test_is_five_prime(self):
        """Is five successfully determined to be prime?"""
        #self.assertTrue(is_prime(5))
        a = np.zeros((5,6,7,4,3))
        self.assertEqual(a[...,0].shape, (5,6,7,4), 'Shape matches')
        
unittest.main()