import keras
from keras import backend as K
from keras import models
from keras.optimizers import SGD
from skimage.io import imread
from keras.callbacks import ModelCheckpoint, LambdaCallback, ReduceLROnPlateau, TensorBoard
import numpy as np
from keras.models import load_model
from functions import your_loss
from process_patches import yield_batch
from models import pawannet

img_rows = 56
img_cols = 56
ysize = 36
n_labels = 4
channels = 3
kernel = 3
crop_size = 10

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
    model = pawannet(img)
    
    reduce_lr = ReduceLROnPlateau(monitor='your_loss', factor=0.2, patience=5, min_lr=0.0001)
    tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    #autoencoder = load_model('models/4160.h5')
    #checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=False)
    #autoencoder.fit(xs, ys, nb_epoch=20, batch_size=64, callbacks=[cb, reduce_lr])
    autoencoder.fit_generator(yield_batch('images/patches/xs/*', 'images/patches/ys/*', 64), samples_per_epoch = 600, nb_epoch = 30, callbacks=[cb, reduce_lr, tbCallBack])
