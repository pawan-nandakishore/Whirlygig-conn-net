import os
import cv2 
import imgaug as ia
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np 
import random as random 
from skimage import io
from skimage import transform as tf
from skimage import color as clr
from PIL import Image 

shear_value = random.randrange(-25,25)
dropout_range = random.uniform(0.0,0.3)
gaussian_range = random.uniform(0.0,2.0)
alpha_for_edge = random.uniform(0.1,1.0)
alpha_for_edge2 = random.uniform(0.1,1.0)
alpha_for_elastic1 =  random.uniform(0.0,1.0)
alpha_for_elastic2 =  random.uniform(0.0,1.0)
random_direction = random.uniform(0.0,1.0)
inversion_value = random.uniform(0,1.0)

aug_seq = iaa.Sequential([
   iaa.Dropout((0.00, dropout_range), per_channel=0.5),
   iaa.Affine(shear=(-shear_value,shear_value),rotate=(-90, 90)),
   iaa.Sometimes(0.5,
            iaa.EdgeDetect(alpha=(0, alpha_for_edge)),
            iaa.DirectedEdgeDetect(alpha=(0,alpha_for_edge2), direction=(0.0, random_direction))),
   iaa.ElasticTransformation(alpha=( alpha_for_elastic1, alpha_for_elastic2), sigma=0.25),          
   iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, gaussian_range), per_channel=0.5),# blur images with a sigma of 0 to 3.0,   
    iaa.Fliplr(0.5),
    iaa.Invert(inversion_value, per_channel=True),
      iaa.Add((-10, 10), per_channel=0.5),
      iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200)) 
] )# do all of the above in random order)
xs_folder = 'patches/xs/'
xs_image_paths = os.listdir(xs_folder)
all_images = [ cv2.imread(xs_folder+xs_image_paths[x])  for x in  range(0, len(xs_image_paths))]

save_folder = 'save_aug/'

if not os.path.isdir(save_folder): 
	os.mkdir(save_folder)
aug_names = ['dropout', 'affine shear', 'edge_detect', 'elastic transform', 'additive gaussian', 'fliplr', 'invert','add','super pixels']
for aug in range(0, len(aug_seq)): 
	augmented_image =aug_seq[aug].augment_image(all_images[5])
	cv2.imwrite(save_folder+aug_names[aug]+'.jpg', augmented_image)








image_num = 5
f, a = plt.subplots(2, sharey= True)
a[0].imshow(all_images[image_num])
a[1].imshow(augmented_images[image_num])
plt.show()