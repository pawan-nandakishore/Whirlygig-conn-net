import os

#os.environ['CUDA_VISIBLE_DEVICES'] = ''

from keras.models import load_model
import matplotlib.pyplot as plt
from skimage import io
from skimage.io import imread
import numpy as np
from skimage.color import gray2rgb
from skimage.io import imsave
from datetime import datetime
from keras import backend as K
from functions import your_loss
import glob
#from scipy.misc import imread
from skimage.io import imread
import shelve 
import glob

#img = io.imread('images/fucked.png')
#gray = io.imread('images/fucked.png', as_grey=True)

# Shelf to sync up

labels = 4
channels = 1
size = 712

#models = ['']
models = glob.glob('models/*')
models.sort(reverse=True)

for model_n in models:
    model = load_model(model_n, custom_objects={'your_loss': your_loss})

    files = glob.glob('raw_images/*')[0:1]
    for idx, fl in enumerate(files):
            print("Processing: %s"%fl)

            img = imread(fl, as_grey=True)
            #img = imread(fl, mode='RGB').astype(float)/255
            #img = imread('images/raw_image_cropped.png', as_grey=True)
            #labels = np.load('labels.npy')
            #labels_432 = np.zeros((432,432,4))
            #labels_432[:-1,:-1,:]=labels
            #print(labels_432.shape)

            # Add the extra row
            xs = np.zeros((size,size))
            xs[:-1,:-1] = img

            xs = xs.reshape(1,channels,size,size)
            print(np.unique(xs[0]))

            result = model.predict(xs).reshape(size,size,labels)

            #assert(xs.max()==1.0)

            zeros = np.zeros((size,size,4))

            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    output = result[i,j]
                    zeros[i,j,np.argmax(output)] = 1
                    #count += 1

            zeros[:,:,3]=1

            #print(count-len(squares2))
            plt.imshow(zeros)
            plt.imsave('plots/%s_%d.png'%(model_n, idx), zeros)
            #plt.imsave('plots/%s_i_%d.png'%(model_n, idx), img)
            #plt.show()
