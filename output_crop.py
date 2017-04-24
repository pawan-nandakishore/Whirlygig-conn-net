import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

from keras.models import load_model
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from skimage.color import gray2rgb
from skimage.io import imsave
from datetime import datetime
from functions import your_loss
import glob
from scipy.misc import imread
#from skimage.io import imread
import glob
import re

#img = io.imread('images/fucked.png')
#gray = io.imread('images/fucked.png', as_grey=True)

# Shelf to sync up

labels = 4
channels = 1
size = 56
inner_size = 36
overlap = 10

def get_tiles(img, inner_size, overlap):
    img_padded = gray2rgb(np.pad(img, ((overlap,overlap), (overlap,overlap)), mode='reflect'))

    xs = []

    for i in xrange(0, img.shape[0], inner_size):
        for j in xrange(0, img.shape[1], inner_size):
            #print(i-overlap+overlap,i+inner_size+overlap+overlap,j-overlap+overlap, j+inner_size+overlap+overlap)
            img_overlapped = img_padded[i:i+inner_size+overlap+overlap,j:j+inner_size+overlap+overlap]
            xs.append(img_overlapped)

    xs = np.array((xs))

#models = ['']
models = sorted(glob.glob('models/*'), key=lambda name: int(re.search(r'\d+', name).group()), reverse=True)
print(models)

for model_n in models:
    model = load_model(model_n, custom_objects={'your_loss': your_loss})

    #files = ['cleaned/raw/1.png']
    #files = glob.glob('cleaned/raw/*')[-2:-1]
    files = glob.glob('cleaned/patches/xs/*')
    print(files)
    for idx, fl in enumerate(files):
            print("Processing: %s"%fl)

            #img = imread(fl, as_grey=True)
            img = imread(fl, mode='L').astype(float)/255
            tiles = get_tiles(img, inner_size, overlap)

            #img = np.invert(imread(fl, mode='L')).astype(float)/255
            #img = imread('images/raw_image_cropped.png', as_grey=True)
            #labels = np.load('labels.npy')
            #labels_432 = np.zeros((432,432,4))
            #labels_432[:-1,:-1,:]=labels
            #print(labels_432.shape)

            # Add the extra row
            #xs = np.zeros((size,size))
            #xs[:-1,:-1] = img
            xs = np.array(tiles)
            count = 0

            xs = xs.reshape(tiles.shape[0],channels,size,size)
            print(np.unique(xs[0]))

            result = model.predict(xs).reshape(inner_size,inner_size,labels)

            assert(xs.max()<=1.0)

            zeros = np.zeros((img.shape[0],img.shape[1],4))

            for i in xrange(0, img.shape[0], inner_size):
                for j in xrange(0, img.shape[1], inner_size):
                    zeros[i:i+inner_size,j:j+inner_size] = result[count]
                    count += 1

            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    output = result[i,j]
                    zeros[i,j,np.argmax(output)] = 1
                    #count += 1

            zeros[:,:,3]=1

            #print(count-len(squares2))
            #plt.imshow(zeros)
            plt.imsave('plots/%s_%d.png'%(model_n, idx), zeros)
            #plt.imsave('plots/%s_i_%d.png'%(model_n, idx), img)
            #plt.show()
