import sys
sys.path.append('..')

import unittest
from unittest2 import TestCase
from skimage.io import imread, imsave
from scipy.misc import imread
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
from scipy.signal import argrelmax
from functions import field_transform
from process_patches import read_data, fetch_batch
import timeit
import glob

class TestDistanceTransform(TestCase):
    """ Numpy slice test """
    def setUp(self):
        print self._testMethodName
        
    def _test_distance_transform(self):
        """ Unit test for pawan's code """
        img = imread('images/0.png')
        plt.imshow(img)
        img_field = field_transform(img)
        plt.figure()
        plt.imshow(img_field)
        
        red, green, blue = img_field[...,0], img_field[...,1], img_field[...,2]
        
        self.assertEqual(red.max(), blue.max(), "Red and blue mountains don't have the same height")
        self.assertEqual(green.max(), 0, "There is no green channel")
        
        # Do kde test. There should be at least 3 unique peaks in the blue kde
        #blue_labeled = label(blue)
        plt.figure()
        plt.hist(blue, bins=5)
        plt.savefig('images/hist.png')
        
        #print(np.unique(blue.flatten()))
        samp = blue.flatten()
        my_pdf = gaussian_kde(samp)
        x = range(256)
        y = my_pdf(x)
        plt.figure()
        plt.plot(x,y,'r') # distribution function
        #plt.hist(samp,normed=1,alpha=.3) # histogram
        plt.savefig('images/kde.png')
        plt.show()

        peaks = argrelmax(y)
        print(peaks)
        self.assertGreater(peaks[0].shape[0], 3, "At least 3 peaks because of the labelling method we have used")
        
        # There should only be one unique peak in the red kde
        
        #print(red.max(), blue.max(), blue_labeled.max())
        
        imsave('images/blue.png', img_field)
        
    def _test_performance(self):
        """ Test on the fly distance transform speed for a batch """
        img = imread('images/0.png')
        
        x, y = read_data(glob.glob('../images/cropped/labeled/*'), glob.glob('../images/cropped/labeled/*'))
        x_patches, y_patches = fetch_batch(x, y, 64, 56, True, True, 20)
        print(x_patches[0].shape)
        
        execution_time = timeit.timeit(lambda: [field_transform(img) for img in x_patches], number=1)
        self.assertLess(execution_time, 1, "Execution time/batch greater than one second")
        
    #def do_field_transform():
        
        
if __name__ == "__main__":
    unittest.main()

img_files = glob.glob('../images/cropped/labeled/*')

for img_fl in img_files:
    img = imread(img_fl, mode='RGB')
    img_dist = field_transform(img)
    plt.figure()
    plt.imshow(img_dist)
    
    imsave(img_fl, img_dist)