from keras import models
from keras.layers import ZeroPadding2D
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from skimage.io import imread
from keras import backend as K
from skimage.transform import rotate
from keras.callbacks import ModelCheckpoint
import numpy as np
import json
from keras.models import load_model

img_w = 432
img_h = 432
n_labels = 3

kernel = 3

autoencoder = models.Sequential()
#autoencoder.add(ZeroPadding2D((1,1), input_shape=(3, img_h, img_w), dim_ordering='th'))

encoding_layers = [
    Convolution2D(32, kernel, kernel, border_mode='same', input_shape=(3, img_h, img_w)),
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

    # Convolution2D(128, kernel, kernel, border_mode='same'),
    # BatchNormalization(),
    # Activation('relu'),
    # Convolution2D(128, kernel, kernel, border_mode='same'),
    # BatchNormalization(),
    # Activation('relu'),
    # MaxPooling2D(),
]

autoencoder.encoding_layers = encoding_layers

for l in autoencoder.encoding_layers:
    autoencoder.add(l)

decoding_layers = [

    # UpSampling2D(),
    # Convolution2D(128, kernel, kernel, border_mode='same'),
    # BatchNormalization(),
    # Activation('relu'),
    # Convolution2D(128, kernel, kernel, border_mode='same'),
    # BatchNormalization(),
    # Activation('relu'),

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
        weights = np.array([ 0.52,  1.3, 0.08])
        #weights = np.array([ 0.05 ,  1.3,  0.55,  4.2])
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


autoencoder.compile(loss=your_loss, optimizer='adam', metrics=['accuracy'])
autoencoder.summary()

#autoencoder = load_model('auto.h5', custom_objects={'your_loss': your_loss})

#autoencoder.save('segnet.h5')

# Time to run the segnet
#img = imread('images/raw_image_cropped2.png', as_grey=True)
#labels = np.load('labels.npy')
#labels_280 = np.zeros((280,280,4))
#labels_280[:-1,:-1,:]=labels
#print(labels_280.shape)

# Add the extra row
#grey = np.zeros((280,280))
#grey[:-1,:-1] = img

#greys = transforms(grey)
#labels = transforms(labels_280)


#xs = np.reshape(greys, (len(greys),1,280,280))
#ys = np.reshape(labels, (len(labels),280*280,4))
xs = np.load('xs_e.npy')
ys = np.load('ys_e.npy')
print(xs.shape, ys.shape)

#def custom_objective(y_true, y_pred):
#    '''Just another crossentropy'''
#y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
#y_pred /= y_pred.sum(axis=-1, keepdims=True)
#cce = T.nnet.categorical_crossentropy(y_pred, y_true)
#    return cce
#
if __name__=="__main__":
    print('lol')
    ##datum = autoencoder.predict(xs, batch_size=1)

    checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=False)
    autoencoder.fit(xs, ys, nb_epoch=10, batch_size=1, callbacks=[checkpointer])

    autoencoder.save('auto.h5')
