

# Create first network with Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
from keras.layers.core import Reshape, Activation
import numpy as np
import random
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
K.set_image_dim_ordering('th')

# Create model
train_data  = np.load('data/xs.npy')
y_vals = np.load('data/ys.npy')
data = zip(train_data, y_vals)
random.shuffle(data)
x, y = zip(*data)
x = np.array(x)
x = x.reshape(x.shape[0], 1, x[0].shape[0], x[0].shape[1])
y = np.array(y)
print(x.shape, x.max(), y.shape, y.max())

# Image data generator, verify that this works
#datagen = ImageDataGenerator(
#        #featurewise_center=True,
#        #featurewise_std_normalization=True,
#        #width_shift_range=0.01,
#        #height_shift_range=0.01,
#        #zoom_range=0.01,
#        rotation_range=90,
#        horizontal_flip=True
#)

cb = keras.callbacks.TensorBoard(log_dir='graph', histogram_freq=10, write_graph=True, write_images=True)

# Create model
model = Sequential()
model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=(1,17,17), name='conv1_1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(16, 3, 3, border_mode='same', name='conv3_1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(16, 3, 3, border_mode='same', name='conv2_1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(16, 3, 3, border_mode='same', name='conv2_3'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x,y,nb_epoch=35, batch_size=32, callbacks=[cb])
#model.fit_generator(datagen.flow(x, y, batch_size=64), samples_per_epoch=len(x), nb_epoch=8)

scores = model.evaluate(x, y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save('models/model.h5')