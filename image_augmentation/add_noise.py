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



xs_folder = 'patches/xs/'
ys_folder = 'patches/ys/'

xs_image_paths = os.listdir(xs_folder)
ys_image_paths = os.listdir(ys_folder)


shear_value = random.randrange(-25,25)
dropout_range = random.uniform(0.0,0.3)
gaussian_range = random.uniform(0.0,2.0)
foreground_seq = iaa.Sequential([ iaa.Affine( shear=( shear_value))]) # shear by -16 to +16 degrees
print(gaussian_range)
# background transform 
aug_seq = iaa.Sequential([
   iaa.Dropout((0.00, dropout_range), per_channel=0.5),
   iaa.Affine(shear=(-shear_value,shear_value),rotate=(-90, 90)),

   iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, gaussian_range), per_channel=0.5),# blur images with a sigma of 0 to 3.0,   
    iaa.Fliplr(0.5) 
], 
random_order=True )# do all of the above in random order)

all_images = [ cv2.imread(xs_folder+xs_image_paths[x])  for x in  range(0, len(xs_image_paths))]

num_range = range(0,len(all_images))
indices  = random.sample(num_range,9 )
index =0
f, mat = plt.subplots(3,3)
for matx in range(0,3):
	for maty in range(0,3):
		show_num = indices[index]
		mat[matx,maty].imshow(all_images[show_num])
		index +=1 

plt.show()

augmented_images =aug_seq.augment_images(all_images)
correct_augmented_images =[]
for x in augmented_images : 
	black_bool = (x[:,:,0] <= 5 ) & ( x[:,:,1] <= 5) & (x[:,:,2] <= 5)
   	
	black_pixels = np.where(black_bool)
	x[black_pixels[0], black_pixels[1]] = 255
	correct_augmented_images.append(x)


augmented_images = correct_augmented_images

dir_name ='augmented_images/'

if not os.path.isdir(dir_name): 
	os.mkdir(dir_name) 

for x in range(0, len(augmented_images)): 
	image = augmented_images[x]
	cv2.imwrite(dir_name+'image_'+str(x)+'.png',image)
	
index = 0


f, mat = plt.subplots(3,3)
for matx in range(0,3):
	for maty in range(0,3):
		show_num = indices[index]
		print(show_num)
		mat[matx,maty].imshow(augmented_images[show_num])
		index +=1 

plt.show()

print len(augmented_images)



