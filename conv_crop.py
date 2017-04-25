import keras
from keras import backend as K
from keras import models
from keras.layers import ZeroPadding2D
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from skimage.io import imread
from keras.callbacks import ModelCheckpoint, LambdaCallback, ReduceLROnPlateau
import numpy as np
from keras.models import load_model
from functions import your_loss

img_w = 56
img_h = 56
ysize=36
n_labels = 4
channels = 1
kernel = 3
crop_size = 10

autoencoder = models.Sequential()
#autoencoder.add(ZeroPadding2D((1,1), input_shape=(3, img_h, img_w), dim_ordering='th'))

encoding_layers = [
    Convolution2D(16, kernel, kernel, border_mode='same', input_shape=(img_h, img_w, channels)),
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
    Convolution2D(n_labels, 1, 1, border_mode='valid'),
    BatchNormalization(),
]
autoencoder.decoding_layers = decoding_layers
for l in autoencoder.decoding_layers:
    autoencoder.add(l)

autoencoder.add(Cropping2D(cropping=((crop_size, crop_size), (crop_size, crop_size))))
autoencoder.add(Reshape((ysize*ysize, n_labels)))
#autoencoder.add(Reshape((n_labels,ysize * ysize)))
#autoencoder.add(Permute((2, 1)))
autoencoder.add(Activation('softmax'))
#autoencoder.add(Permute((1, 2)))
autoencoder.add(Reshape((ysize,ysize,n_labels)))

#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#rmsprop = keras.optimizers.RMSprop(lr=1e-5, rho=0.9, epsilon=1e-08, decay=0.0)
autoencoder.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#autoencoder.compile(loss=your_loss, optimizer='adam', metrics=['accuracy'])
autoencoder.summary()


xs = np.load('xs_s.npy')
ys = np.load('ys_s.npy')
#xs = np.load('data/xs.npy')
#ys = np.load('data/ys.npy')

print(xs.shape, ys.shape)

def save_mod(epoch, logs):
    global count
    global autoencoder
    if count%40==0:
        print('Saving model, count: %d'%count)
        autoencoder.save('models/%d.h5'%count)
    count+=1

count = 0
cb = LambdaCallback(on_batch_begin=save_mod)

if __name__=="__main__":
    print('lol')

    reduce_lr = ReduceLROnPlateau(monitor='your_loss', factor=0.2, patience=5, min_lr=0.0001)
    #autoencoder = load_model('models/3980.h5', custom_objects={'your_loss': your_loss})
    #checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=False)
    autoencoder.fit(xs, ys, nb_epoch=20, batch_size=64, callbacks=[cb, reduce_lr])

