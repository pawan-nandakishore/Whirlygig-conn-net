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
from functions import your_loss
import glob
from scipy.misc import imread
#from skimage.io import imread
import glob
import re
import time

labels = 4
channels = 1
size = 1080
batch_size = 12

def output_to_colors(result, x):
    #zeros = np.zeros((rows,cols,4))
    #zeros[:,:,:-1]=gray2rgb(x.copy())
    zeros = gray2rgb(x.copy())
    output = result.argmax(axis=-1)
    zeros[output==2]=[0,0,1]
    return zeros

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


models = sorted(glob.glob('models/*'), key=lambda name: int(re.search(r'\d+', name).group()), reverse=True)[0:1]
print(models)

for model_n in models:
    model = load_model(model_n, custom_objects={'your_loss': your_loss})
    print("Loaded :%s", model_n)

    files_all = glob.glob('images/single_frames/*.png')
    #files_all = glob.glob('images/single_frames/*')
    #files = files+files+files# + files[-4:-1]
    #print(files)

    file_chunks = chunks(files_all, batch_size)

    for idx, files in enumerate(file_chunks):
        file_names = [basename(path) for path in files]
        print(file_names)
        imgs = np.array([imread(fl, mode='L').astype(float)/255 for fl in files])
        #print(file_chunks)
        #print("Processing: %s"%(fl))
        print("Imgs shape: %s", imgs.shape)

        #Create input tensor
        xs = imgs.reshape(imgs.shape[0],channels,size,size)
        print(np.unique(xs[0]))

        start_time = time.time()

        # Predict output
        ys = model.predict(xs)
        print("---- %s seconds for size: %d ----"%(time.time()-start_time, xs.shape[0]))
        ys = ys.reshape(xs.shape[0], size, size, labels)

        colors = [output_to_colors(y, imgs[i]) for i,y in enumerate(ys)]
        #colors = [label2rgb(y.argmax(axis=-1), image=imgs[i], colors=[(1,0,0), (0,1,0), (0,0,1), (0,0,0)], alpha=0.9, bg_label=3) for i,y in enumerate(ys)]

        [plt.imsave('plots/%s_%s'%(model_n, file_names[i]), zeros) for i,zeros in enumerate(colors)]
