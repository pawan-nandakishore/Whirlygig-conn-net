import numpy as np
import glob
from scipy.misc import imread
from skimage.io import imsave
from skimage.transform import rotate
import matplotlib.pyplot as plt
import os
from os.path import basename
import glob
import random
from functions import transforms, raw_to_labels
import elastic_pawan_ver as elastic 
import sys 

print("the import works ! ")

# This is a comment. Move along people.

height = 56
width = 56
yheight = 36
ywidth = 36
n_labels = 4
n_channels = 3

img_names = sorted(glob.glob('cleaned/patches/xs/*'))
label_names = sorted(glob.glob('cleaned/patches/ys/*'))

imgs = [imread(fl, mode='RGB') for fl in img_names]
labels = [imread(fl, mode='RGB') for fl in label_names]

raw_deformed_images  = []
labeled_deformed_images  = []
print("imgs length/ labels", len(imgs), len(labels))
for im in range(0,len(imgs)): 

	labeled_aug, raw_aug = elastic.apply_deformation(imgs[im],labels[im] )

	raw_deformed_images.append(raw_aug)
	labeled_deformed_images.append(labeled_aug)
	# print('raw aug:', raw_aug[0].shape)
	


raw_deformed_images = [x for y in raw_deformed_images for x in y]
labeled_deformed_images = [x for y in labeled_deformed_images for x in y]



# print(len(raw_deformed_images),len(labeled_deformed_images) )


xs = []
ys = []

imgs2 = raw_deformed_images
labels2  = labeled_deformed_images


# plt_indx = 0 
# f, a = plt.subplots(3,3)
# for xp in range(0,3):
# 	for yp in range(0,3):
# 		a[xp,yp].imshow(imgs[plt_indx])
# 		plt_indx+=1 
# plt.show()



count = 1

for img, out in zip(imgs, labels):
    x = img
    #x = np.invert(x)
    y = raw_to_labels(out, count)

    xs.extend(transforms(x))
    ys.extend(transforms(y))

    count += 1

# Shuffle the data
data = zip(xs, ys)
random.shuffle(data)
xs, ys = zip(*data)

# Convert to numpy array
xs = np.array(xs)
ys = np.array(ys)
print(xs.shape, ys.shape)

# Reshape: xs: num, labels, size, size,  ys: num, size*size, labels
xs = xs.reshape(xs.shape[0], height, width, n_channels).astype(float)/255 # Convert to float between 0-1
#xs = xs.reshape(xs.shape[0], n_channels, height, width).astype(float)/255 # Convert to float between 0-1
ys = ys.reshape(xs.shape[0], yheight, ywidth, n_labels).astype(float) # Convert to one hot float between 0-1

# Some descriptive statistics
print(xs.shape, ys.shape, np.unique(xs), np.unique(ys))

np.save('xs_s.npy',xs) # Normalize between 0-1
np.save('ys_s.npy',ys)
