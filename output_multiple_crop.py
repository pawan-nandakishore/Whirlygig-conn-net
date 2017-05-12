import os
from os.path import basename

#os.environ['CUDA_VISIBLE_DEVICES'] = ''

from keras.models import load_model
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from skimage.color import gray2rgb, label2rgb
from skimage.io import imsave
from datetime import datetime
from functions import your_loss, squares_to_tiles, labels_to_raw, tiles_to_square
import glob
from scipy.misc import imread
#from skimage.io import imread
import glob
import re
import time
import random
import cv2

labels = 4
channels = 3
size = 56
inner_size = 36
ysize = 36
batch_size = 2

def timeit(lambdaFunc):
    """ Runs lambdaFunc and prints the execution time as well """
    start_time = time.time()
    result= lambdaFunc()
    print("---- %s seconds for size: %d ----"%(time.time()-start_time, xs.shape[0]))
    return result

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

if __name__ == "__main__":

    models = sorted(glob.glob('models/*'), key=lambda name: int(re.search(r'\d+', name).group()), reverse=True)[0:1]
    print(models)
    
    for model_n in models:
        model = load_model(model_n, custom_objects={'your_loss': your_loss})
        print("Loaded :%s", model_n)
    
        files_all = glob.glob('images/cropped/rgbs/*.png')
    
        file_chunks = chunks(files_all, batch_size)
    
        # Predict output
        for idx, files in enumerate(file_chunks):
            # Read data
            file_names = [basename(path) for path in files]
            imgs = [imread(fl, mode='RGB').astype(float)/255 for fl in files]
            imgs = np.array([np.pad(img, ((10,10), (10,10), (0,0)), mode='reflect') for img in imgs])
            
            # Generate tiles
            tiles = np.array([squares_to_tiles(img, (56,56), (36,36)) for img in imgs])
            print("Imgs shape: %s", tiles.shape)
    
            # Input and output calculation
            xs = tiles.reshape(imgs.shape[0]*len(tiles[0]),size,size,channels)
            ys = timeit(lambda: model.predict(xs))
            
            # Reshape and convert patches to rgbs
            ys = ys.reshape(imgs.shape[0],len(tiles[0]), ysize, ysize, labels)
            ys = np.array([labels_to_raw(y) for y in ys])            
            
            # Patches to bigger image and save
            y_images = [tiles_to_square(patches, (288,288,3), (36,36,3), (36,36,3)) for patches in ys]
            [imsave('plots/results/%s'%(file_name), y_img/255) for file_name, y_img in zip(file_names, y_images)]
            
            
