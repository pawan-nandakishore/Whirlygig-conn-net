#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 01:46:36 2017

@author: monisha
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import glob
from models import pawannet_autoencoder
from keras.callbacks import LambdaCallback
from keras.models import load_model
from keras.callbacks import History
from process_patches import tensor_blur, crop_tensor, sample_patches, read_data

class PawannetDistTransform():
    """ Testing pawannet autoencoder on whirlygig images """
    """ Switch from 3 channels to 2. Last channel is unnecessary """
    
    def __init__(self):
        self.count = 0
        self.model = pawannet_autoencoder((56,56,3), (36,36,3), 10, kernel=3)
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        self.cb = LambdaCallback(on_batch_begin=self.save_mod)
        
    def preprocess(self, x_patches, y_patches, crop_size):
        """ Take twp input tensors x and y and apply some lambda augmentations on them """
        
        y_patches = crop_tensor(y_patches, crop_size) #Crop y for overlapping tile strategy
        x_patches = tensor_blur(np.uint8(x_patches)) # Apply median blur to reduce neck variation, tensorize this
        
        return x_patches.astype(float)/255, y_patches.astype(float)/255 # Why is this necessary?
    
    def yield_batch(self, x, y, n=64, patch_size=56, crop_size=20):  
        """ Yields batch of size n infinitely """
        
        while True:
            x_patches, y_patches = sample_patches(x, y, n, (1, patch_size, patch_size, 3))
        
            x_aug, y_aug = self.preprocess(x_patches, y_patches, crop_size)
            
            yield (x_aug, y_aug)
            
    def save_mod(self, epoch, logs):
        if self.count%40==0:
            print('Saving model, count: %d'%self.count)
            self.model.save('models/%d.h5'%self.count)
        self.count+=1
        
    def run(self):
        """ Train network on whirlygig images and check that the training loss converges """
        history = History()
        x, y = read_data(glob.glob('images/cropped/rgbs/*'), glob.glob('images/cropped/labeled/*'))
        
        # Testing by visualizing
        #x_test,y_test = fetch_batch(x, y, n=10, patch_size=56, preprocess=False, augment=True, crop_size=20)
        #[imsave('%d_i.png'%i, img.astype(float)/255) for i,img in enumerate(x_test)]
        #[imsave('%d_o.png'%i, img.astype(float)/255) for i,img in enumerate(y_test)]
        
        # Run the network
        dataGenerator = self.yield_batch(x, y, n=64, patch_size=56, crop_size=20)
        self.model.fit_generator(dataGenerator, samples_per_epoch = 600, nb_epoch = 20, callbacks=[self.cb, history])
        
        self.assertGreater(history.history['val_acc'][0], 0.4)
        
    def sanity_checks(self):
        """ Test that the model follows some particular distances """
        x, y = read_data(glob.glob('images/cropped/rgbs/*'), glob.glob('images/cropped/labeled/*'))
        x_patches, y_patches = sample_patches(x, y, 64, (1, 56, 56, 3))
        print(x_patches.shape, y_patches.shape)
        print(np.unique(x_patches), np.unique(y_patches), np.max(x_patches), np.max(y_patches))
        
        x_aug, y_aug = self.preprocess(x_patches, y_patches, 20)
        print(x_aug.shape, y_aug.shape)
        #print(np.unique(x_aug[0]), np.unique(y_aug[0]), np.max(x_aug[0]), np.max(y_aug[0]))
        
        plt.imshow(x_aug[0])
        plt.figure()
        plt.imshow(y_aug[0])
        
        np.testing.assert_almost_equal(np.max(x_patches), 255.0)
        np.testing.assert_almost_equal(np.max(x_aug), 1.0)
        np.testing.assert_equal(x_aug.shape, (64,56,56,3))
        np.testing.assert_equal(y_aug.shape, (64,36,36,3))
        
    def visualize(self):
        """ Add code to visualize this particular network right now """
        # Run gradcams on the network after training processing is over. Move visualize code to runner class
        pass
        
if __name__ == "__main__":
    runner = PawannetDistTransform()
    runner.sanity_checks()
    runner.run()
