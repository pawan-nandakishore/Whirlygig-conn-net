
# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import random
from keras import backend as K

# Load data
train_data  = np.load('training_data.npy')
y_vals = np.load('y_vals.npy')
data = zip(train_data, y_vals)
random.shuffle(data)
x, y = zip(*data)
x = np.array(x)
y = np.array(y)

# Create model
model = Sequential()
model.add(Dense(50, input_dim=25, init='uniform', activation='relu'))
model.add(Dense(20, init='uniform', activation='relu'))
model.add(Dense(3, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(x, y, nb_epoch=15, batch_size=10)
# evaluate the model
scores = model.evaluate(x, y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save('full.h5')
