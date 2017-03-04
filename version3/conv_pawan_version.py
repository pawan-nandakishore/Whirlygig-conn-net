

# Create first network with Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
from keras.layers.core import Reshape, Activation
import numpy as np
import random
from keras import backend as K
import sys
from keras.layers.normalization import BatchNormalization

K.set_image_dim_ordering('th')

# Create model
train_data  = np.load('x_train.npy')
y_vals = np.load('y_train.npy')
print(train_data.shape)
train_data = train_data/255.0
window_size =6

number_of_features1 =16
number_of_features2 =16
number_of_features3 = 16
number_of_features4 = 16
number_of_features5 = 16

number_of_features_dense = 64
# Create model
model = Sequential()
model.add(Convolution2D(number_of_features1, 3, 3, border_mode='same', input_shape=(1,window_size*2,window_size*2), name='conv1_1'))
model.add(Activation('relu'))
 # model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='conv1_2'))
model.add(Convolution2D(number_of_features2, 3, 3, activation='relu', border_mode='same', name='conv2_1'))
model.add(BatchNormalization())
model.add(Convolution2D(number_of_features3, 3, 3, activation='relu', border_mode='same', name='conv3_1'))
model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
model.add(Convolution2D(number_of_features4, 3, 3, activation='relu', border_mode='same', name='conv4_1'))
model.add(BatchNormalization())
model.add(Convolution2D(number_of_features4, 3, 3, activation='relu', border_mode='same', name='conv5_1'))
model.add(BatchNormalization())
# model.add(Convolution2D(number_of_features5, 3, 3, activation='relu', border_mode='same', name='conv6_1'))
# model.add(BatchNormalization())


model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

model.add(Flatten())
model.add(Dense(number_of_features_dense, activation='relu'))
model.add(Dense(number_of_features_dense, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# keras.optimizers.SGD(lr=0.000002, momentum=1e-4, decay=0.0, nesterov=True)
# keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_data,y_vals,nb_epoch=20, batch_size=50)

scores = model.evaluate(train_data, y_vals)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save('conv.h5')
