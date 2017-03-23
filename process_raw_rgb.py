import numpy as np
import glob
from scipy.misc import imread
from skimage.io import imsave
from skimage.transform import rotate
import matplotlib.pyplot as plt
import os
import cv2
import glob

def rotate_thrice(square):
        return [square, np.rot90(square, 1), np.rot90(square, 2), np.rot90(square, 3)]

def transforms(square):
        return rotate_thrice(square) + rotate_thrice(np.fliplr(square))

def raw_to_labels(image):
    assert(image.max()==255)
    inside_bool = (image[:,:,0] <= 120 ) & ( image[:,:,1] <= 120) & (image[:,:,2] >= 130 )
    boundary_bool = (image[:,:,0] <=120  ) & ( image[:,:,1] >= 150) & (image[:,:,2] <= 120 )
    exterior_bool = ~inside_bool & ~boundary_bool

    softmax_labeled_image = np.zeros(image.shape)
    softmax_labeled_image[inside_bool] = [1,0,0]
    softmax_labeled_image[boundary_bool] = [0,1,0]
    softmax_labeled_image[exterior_bool] = [0,0,1]
    return softmax_labeled_image


size = 432
n_labels = 3

img_names = sorted(glob.glob('cropped/*'))
label_names = sorted(glob.glob('labeled_cropped/*'))

imgs = [imread(fl, mode='RGB') for fl in img_names]
labels = [imread(fl, mode='RGB') for fl in label_names]

zeros = np.zeros((size, size, 4))

xs = []
ys = []

for img, out in zip(imgs, labels):
    x = img
    y = raw_to_labels(out)

    xs.extend(transforms(x))
    ys.extend(transforms(y))

xs = np.array(xs)
ys = np.array(ys)

xs = xs.reshape(xs.shape[0], n_labels, size, size).astype(float)/255 # Convert to float between 0-1
ys = ys.reshape(xs.shape[0], size*size, n_labels).astype(float) # Convert to one hot float between 0-1
print(xs.shape, ys.shape)

np.save('xs_e.npy',xs) # Normalize between 0-1
np.save('ys_e.npy',ys)
