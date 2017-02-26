

# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
from keras.layers.core import Reshape, Activation
import numpy as np
import random
from keras import backend as K
K.set_image_dim_ordering('th')

# Create model
train_data  = np.load('xs.npy')
y_vals = np.load('ys.npy')
data = zip(train_data, y_vals)
random.shuffle(data)
x, y = zip(*data)
x = np.array(x)
x = x.reshape(x.shape[0], 1, x[0].shape[0], x[0].shape[1])
y = np.array(y)
print(x.shape, y.shape)

# Create model
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(1,31,31), name='conv1_1'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='conv2_1'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x,y,nb_epoch=15, batch_size=32)

scores = model.evaluate(x, y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save('conv.h5')
