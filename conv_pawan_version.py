

# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
from keras.layers.core import Reshape, Activation
import numpy as np
import random
from keras import backend as K
import sys
K.set_image_dim_ordering('th')

# Create model
train_data  = np.load('x_train.npy')
y_vals = np.load('y_train.npy')
print(train_data.shape)

window_size =3


# Create model
model = Sequential()
model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(1,window_size*2,window_size*2), name='conv1_1'))
model.add(Activation('relu'))
# model.add(Convolution2D(8, 3, 3, activation='relu', border_mode='same', name='conv1_2'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='conv2_1'))
# model.add(Convolution2D(8, 3, 3, activation='relu', border_mode='same', name='conv2_2'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_data,y_vals,nb_epoch=120, batch_size=200)

scores = model.evaluate(train_data, y_vals)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save('conv.h5')
