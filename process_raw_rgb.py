import numpy as np
import glob
from scipy.misc import imread
from skimage.io import imsave
from skimage.transform import rotate
import matplotlib.pyplot as plt
import os
import glob
import random
from functions import transforms, raw_to_labels


height = 1080
width = 1080
n_labels = 4
n_channels = 1

img_names = sorted(glob.glob('cleaned/raw/*'))
label_names = sorted(glob.glob('cleaned/labeled/*'))

imgs = [imread(fl, mode='L') for fl in img_names]
labels = [imread(fl, mode='RGB') for fl in label_names]


xs = []
ys = []

for img, out in zip(imgs, labels):
    x = img
    #x = np.invert(x)
    y = raw_to_labels(out)

    xs.extend(transforms(x))
    ys.extend(transforms(y))

# Shuffle the data
data = zip(xs, ys)
random.shuffle(data)
xs, ys = zip(*data)

# Convert to numpy array
xs = np.array(xs)
ys = np.array(ys)

# Reshape: xs: num, labels, size, size,  ys: num, size*size, labels
xs = xs.reshape(xs.shape[0], n_channels, height, width).astype(float)/255 # Convert to float between 0-1
ys = ys.reshape(xs.shape[0], height*width, n_labels).astype(float) # Convert to one hot float between 0-1

# Some descriptive statistics
print(xs.shape, ys.shape, np.unique(xs), np.unique(ys))

np.save('xs_s.npy',xs) # Normalize between 0-1
np.save('ys_s.npy',ys)
