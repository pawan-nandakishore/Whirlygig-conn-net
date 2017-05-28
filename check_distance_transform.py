import numpy as np
import glob
from scipy.misc import imread
import random
from functions import sort_by_number, raw_to_labels, plot_row, sample_patches
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import  process_patches as pp
from scipy import ndimage
 
    

#This code is to test how to apply the distance transform to each image of the patches..
# The idea is to apply the distance transform to the whole image and then take the patches


x, y = pp.read_data(glob.glob('images/cropped/rgbs/*'), glob.glob('images/cropped/labeled/*'))





def get_dist_trans(input_image) :
	"""
	This function calculates the distance transform for a given whirlygig image
	and primarily outputs the distance transformed for the image
	
	Input : A labeled whirlygig image, type : N*N rgb image with class labels

	Output : A list consisting of the following items in the given order  
	
	1) Distance transform of the dilated image which in turn is acquired by 
		dilating the rgb input image 

	2) Dilated image of the original image	

	3) individual beetle image, this is a (nb, N,N) sized image where nb is 
		total number of beetles. Each image is a binary image where the 
		blue region of the labeled image is assigned as 255 and the background 
		is assigned as 0 

	4) Dilated then distance transformed image of the individual beetle images. Is as 
		(nb,N,N) shaped array 

	the function also outputs a plot comparing an image of dilated whirlygigs to that of 
	a distance transformed image of the dilated image of whirlygigs
	"""

	# this is the input image 
	single_image = input_image


	# pull out the green pixels from the green channels
	green_pixelsx,green_pixelsy = np.where(single_image[:,:,1]==255)  
	
	#dilating the image then taking the distance transform
	red_labeled_image, red_nfeatures = ndimage.label(single_image[:,:,1])
	 
	# get a labeled image, this assigns a label to each of the whirlygigs  
	labeled_image, nfeatures = ndimage.label(single_image[:,:,2])
	all_objects = ndimage.find_objects(labeled_image)

	# initialization for a variety of objects that will be used in the code below
	individual_beetles = np.zeros((nfeatures,labeled_image.shape[1], labeled_image.shape[1]))
	individual_beetles_dilated = []
	individual_dilated_distrans = []
	total_background = np.zeros((labeled_image.shape[1], labeled_image.shape[1]))
	blue_distrans_background = np.zeros((labeled_image.shape[1], labeled_image.shape[1]))
	new_image = np.zeros(single_image.shape, 'uint8')
	green_channel = np.zeros((labeled_image.shape[0], labeled_image.shape[0]))
	

	#get the dilated and the distance transformed image for each of the beetles images
	for bet in range(1,nfeatures):

		#get the pixels for the each of the whirlgigs. 
		betx,bety =  np.where(labeled_image==bet)

		# the variable individual_beetles is a (nb,N,N) image where nb is the index of a 
		# new whirlygig 
		individual_beetles[bet,betx,bety] = 255
	
		# 
		ind_dilated = ndimage.binary_dilation(individual_beetles[bet,:,:], iterations = 2)
		ind_dilated_distrans= ndimage.distance_transform_edt(ind_dilated)
		
		individual_beetles_dilated.append(ind_dilated)
		individual_dilated_distrans.append(ind_dilated_distrans)
		
		total_background +=ind_dilated
		blue_distrans_background += ind_dilated_distrans
		


	scalex, scaley = np.where(total_background>1)
	blue_distrans_background[scalex,scaley] = blue_distrans_background[scalex,scaley]/total_background[scalex,scaley] 

	individual_beetles_dilated = np.array(individual_beetles_dilated)
	individual_dilated_distrans = np.array(individual_dilated_distrans)


	blue_disttrans_image = (blue_distrans_background/np.max(blue_distrans_background))*255 
	
	# plt.imshow()
	# plt.show()

	# print("labeled image :",single_image.shape)
	new_image[:,:,2] = blue_disttrans_image.reshape(blue_disttrans_image.shape[0], blue_disttrans_image.shape[0])
	new_image[:,:,1] = green_channel     	
	new_image[:,:,0] = single_image[:,:,0]
	new_image = Image.fromarray(new_image)
	# blues are 0,0,255 greens are 0,255,0 
	all_data= [blue_distrans_background,total_background, individual_beetles, individual_dilated_distrans, new_image]

	# f,ax = plt.subplots(2, sharey =True)
	# ax[0].imshow(distrans_background)
	# ax[1].imshow(total_background )
	# plt.show()




	return all_data



print(y[0].shape)
all_data = get_dist_trans(y[0])

dist_transformed_image = all_data[0]
new_labeled_image = all_data[4]




plt.imshow(new_labeled_image)
plt.show()
plt.imsave( 'dist_transformed_image_labeled.jpg',new_labeled_image)