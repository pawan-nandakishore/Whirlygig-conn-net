from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave
from scipy.misc import imread
from skimage.color import gray2rgb
import glob
from functions import sort_by_number

img_files = sort_by_number(glob.glob('images/rgbs/*'))
label_files = sort_by_number(glob.glob('images/labeled/*.png'))
crops = [(380,532), (371,540), (416,525), (439,465), (386,502)]

def get_crop(img, crop, square=300):
    """ Returns a croppped square """
    return img[crop[1]:crop[1]+square,crop[0]:crop[0]+square]

imgs_cropped = [get_crop(imread(img_fl), crop) for crop, img_fl in zip(crops, img_files)]
labels_cropped = [get_crop(imread(label_fl), crop) for crop, label_fl in zip(crops, label_files)]

# Let's solve this toy problem first
for i, (img_c, label_c) in enumerate(zip(imgs_cropped, labels_cropped)):
    imsave('images/cropped/rgbs/%d.png'%i, img_c)
    imsave('images/cropped/labeled/%d.png'%i, label_c)