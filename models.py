from keras import models
from keras.layers import ZeroPadding2D, Input
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D as Conv2D
from keras.layers import concatenate
from keras.models import Model

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 04:40:42 2017

@author: monisha
"""

def pawannet(input_shape, output_shape, crop_size, kernel=3):
    """ Does not work with autoencoding task. Modified segnet with lesser filters. How to add tests for these? Can we add performance tests? """
    autoencoder = models.Sequential()
    #autoencoder.add(ZeroPadding2D((1,1), input_shape=(3, img_h, img_w), dim_ordering='th'))
    
    encoding_layers = [
        Convolution2D(16, kernel, kernel, border_mode='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(16, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
    
        Convolution2D(32, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(32, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
    ]
    
    autoencoder.encoding_layers = encoding_layers
    
    for l in autoencoder.encoding_layers:
        autoencoder.add(l)
    
    decoding_layers = [
    
        UpSampling2D(),
        Convolution2D(32, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(32, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
    
        UpSampling2D(),
        Convolution2D(16, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(output_shape[-1], 1, 1, border_mode='valid'),
        BatchNormalization(),
    ]
    autoencoder.decoding_layers = decoding_layers
    for l in autoencoder.decoding_layers:
        autoencoder.add(l)
    
    autoencoder.add(Cropping2D(cropping=((crop_size, crop_size), (crop_size, crop_size))))
    autoencoder.add(Reshape((output_shape[0]*output_shape[1], output_shape[-1])))
    #autoencoder.add(Reshape((n_labels,ysize * ysize)))
    #autoencoder.add(Permute((2, 1)))
    autoencoder.add(Activation('softmax'))
    #autoencoder.add(Permute((1, 2)))
    autoencoder.add(Reshape(output_shape))
    
    #sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    #rmsprop = keras.optimizers.RMSprop(lr=1e-5, rho=0.9, epsilon=1e-08, decay=0.0)
    return autoencoder

def pawannet_autoencoder(input_shape, output_shape, crop_size, kernel=3):
    """ Autoencoder version of pawannet """
    autoencoder = models.Sequential()
    #autoencoder.add(ZeroPadding2D((1,1), input_shape=(3, img_h, img_w), dim_ordering='th'))
    
    encoding_layers = [
        Convolution2D(16, kernel, kernel, border_mode='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(16, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
    
        Convolution2D(8, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(8, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
    ]
    
    autoencoder.encoding_layers = encoding_layers
    
    for l in autoencoder.encoding_layers:
        autoencoder.add(l)
    
    decoding_layers = [
    
        UpSampling2D(),
        Convolution2D(8, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(8, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
    
        UpSampling2D(),
        Convolution2D(16, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(output_shape[-1], 1, 1, border_mode='valid'),
        #BatchNormalization(),
    ]
    autoencoder.decoding_layers = decoding_layers
    for l in autoencoder.decoding_layers:
        autoencoder.add(l)
    
    autoencoder.add(Cropping2D(cropping=((crop_size, crop_size), (crop_size, crop_size))))
    autoencoder.add(Reshape((output_shape[0]*output_shape[1], output_shape[-1])))
    autoencoder.add(Activation('sigmoid'))
    autoencoder.add(Reshape(output_shape))
    
    return autoencoder

def unet(input_shape, output_shape, crop_size, kernel=3):
    """ Unet implementation. Test performance on mnist autoencoding task. """
    inputs = Input(input_shape)
    conv1 = Conv2D(32, (kernel, kernel), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (kernel, kernel), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (kernel, kernel), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (kernel, kernel), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (kernel, kernel), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (kernel, kernel), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (kernel, kernel), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (kernel, kernel), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (kernel, kernel), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (kernel, kernel), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (kernel, kernel), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (kernel, kernel), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (kernel, kernel), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (kernel, kernel), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (kernel, kernel), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (kernel, kernel), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (kernel, kernel), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (kernel, kernel), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(output_shape[-1], (1, 1), activation='sigmoid')(conv9)
    
    crop1 = Cropping2D(cropping=((crop_size, crop_size), (crop_size, crop_size)))(conv10)
    reshape1 = Reshape((output_shape[0]*output_shape[1], output_shape[-1]))(crop1)
    
    sigmoid1 = Activation('sigmoid')(reshape1)
    #reshape2 = Reshape(output_shape)
    
    model = Model(inputs=[inputs], outputs=[sigmoid1])

    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def residual_network(img_shape):
    """ Test residual network with skip connections """
    pass

def keras_mnist_autoencoder(img_shape, output_shape, crop_size, kernel=3):
    """ Copy pasted autoencoder from keras to use as a benchmark. """
    input_img = Input(img_shape)
    x = Conv2D(16, (kernel, kernel), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (kernel, kernel), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (kernel, kernel), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    
    x = Conv2D(32, (kernel, kernel), activation='relu', padding='same')(encoded)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (kernel, kernel), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (kernel, kernel), activation='relu')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (kernel, kernel), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    print('lol')
    return autoencoder

def dilated_convolution(input_shape, output_shape, crop_size, kernel=3):
    """ Fixed image resolution but exponentially increasing receptive field size """
    print('lol2')
    pass