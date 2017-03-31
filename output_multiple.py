import os

#os.environ['CUDA_VISIBLE_DEVICES'] = ''

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
import time

labels = 4
channels = 1
size = 1080

def output_to_colors(result, rows, cols, x):
    zeros = np.zeros((rows,cols,4))
    zeros[:,:,:-1]=gray2rgb(x.copy())

    for i in range(rows):
	for j in range(cols):
	    output = result[i,j]
            index = np.argmax(output)
            if index == 2:
	        zeros[i,j,index] = 1
	    #count += 1

    zeros[:,:,3]=1
    return zeros


models = sorted(glob.glob('models/*'), key=lambda name: int(re.search(r'\d+', name).group()), reverse=True)[0:1]
print(models)

for model_n in models:
    model = load_model(model_n, custom_objects={'your_loss': your_loss})
    print("Loaded :%s", model_n)

    files = glob.glob('cleaned/raw/*')
    files = files + files[-4:-1]
    print(files)

    imgs = np.array([imread(fl, mode='L').astype(float)/255 for fl in files])
    print("Processing: %s"%fl)
    print("Imgs shape: %s", imgs.shape)

    #Create input tensor
    xs = imgs.reshape(imgs.shape[0],channels,size,size)
    print(np.unique(xs[0]))

    start_time = time.time()

    # Predict output
    ys = model.predict(xs)
    print("---- %s seconds for size: %d ----"%(time.time()-start_time, xs.shape[0]))
    ys = ys.reshape(xs.shape[0], size, size, labels)

    colors = [output_to_colors(y, size, size, imgs[i]) for i,y in enumerate(ys)]

    [plt.imsave('/home/thutupallilab/Dropbox/Whirlygig/plots/%s_%d.png'%(model_n, idx), zeros) for idx,zeros in enumerate(colors)]
