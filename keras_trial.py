

# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]


train_data  = np.load('training_data.npy')
y_vals = np.load('y_vals.npy') 


#train_data  = np.load('training_data.npy')
#y_vals = np.load('y_vals.npy') 

# create model
model = Sequential()
model.add(Dense(50, input_dim=25, init='uniform', activation='relu'))
model.add(Dense(20, init='uniform', activation='relu'))
model.add(Dense(4, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(train_data, y_vals, nb_epoch=15, batch_size=10)
# evaluate the model
scores = model.evaluate(train_data, y_vals)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))