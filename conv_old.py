import keras
from keras import backend as K
from keras import models
from keras.layers import ZeroPadding2D
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from skimage.io import imread
from keras.callbacks import ModelCheckpoint, LambdaCallback
import numpy as np
from keras.models import load_model

img_w = 712
img_h = 712
n_labels = 4
channels = 1
kernel = 3

autoencoder = models.Sequential()
#autoencoder.add(ZeroPadding2D((1,1), input_shape=(3, img_h, img_w), dim_ordering='th'))

encoding_layers = [
    Convolution2D(32, kernel, kernel, border_mode='same', input_shape=(channels, img_h, img_w)),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(32, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),

    Convolution2D(64, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(64, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),
]

autoencoder.encoding_layers = encoding_layers

for l in autoencoder.encoding_layers:
    autoencoder.add(l)

decoding_layers = [

    UpSampling2D(),
    Convolution2D(64, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(64, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(),
    Convolution2D(32, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(n_labels, 1, 1, border_mode='valid'),
    BatchNormalization(),
]
autoencoder.decoding_layers = decoding_layers
for l in autoencoder.decoding_layers:
    autoencoder.add(l)

autoencoder.add(Reshape((n_labels, img_h * img_w)))
autoencoder.add(Permute((2, 1)))
autoencoder.add(Activation('softmax'))


def your_loss(y_true, y_pred):
        #weights = np.ones(4)
        #weights = np.array([ 1.5,  0.8, 0.008])
        #weights = np.array([0.99524712791495196, 0.98911715534979427, 0.015705375514403319])
        #weights = np.array([ 0.91640706, 0.60022308, 0.001442506])
        weights = np.array([ 4.2 ,  0.52,  1.3,  0.08])
        #weights = np.array([0.00713773, 0.20517703, 0.15813273, 0.62955252])
        #weights = np.array([1,,0.1,0.001])
        # scale preds so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip
        y_pred = K.clip(y_pred, K.epsilon(), 1)
        # calc
        loss = y_true*K.log(y_pred)*weights
        loss =-K.sum(loss,-1)
        return loss

def rotate_thrice(square):
        return [square, rotate(square, 90), rotate(square, 180), rotate(square, 270)]

def transforms(square):
        return rotate_thrice(square) + rotate_thrice(np.fliplr(square))


#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#rmsprop = keras.optimizers.RMSprop(lr=1e-5, rho=0.9, epsilon=1e-08, decay=0.0)
autoencoder.compile(loss=your_loss, optimizer='adam', metrics=['accuracy'])
autoencoder.summary()


#xs = np.load('xs_e.npy')
#ys = np.load('ys_e.npy')
xs = np.load('data/xs.npy')
ys = np.load('data/ys.npy')

print(xs.shape, ys.shape)

def save_mod(epoch, logs):
    global count
    global autoencoder
    if count%5==0:
        print('Saving model, count: %d'%count)
        autoencoder.save('models/%d.h5'%count)
    count+=1

count = 0
cb = LambdaCallback(on_batch_begin=save_mod)

if __name__=="__main__":
    print('lol')

    #auto = load_model('auto.h5', custom_objects={'your_loss': your_loss})
    #checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=False)
    autoencoder.fit(xs, ys, nb_epoch=10, batch_size=1, callbacks=[cb])

    autoencoder.save('auto.h5')
