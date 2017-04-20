
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



def augment_images(foreground_seq, background_seq, single_image, single_raw_image, show_plots =True):
	
	
	if(single_image.shape != single_raw_image.shape): 
		print('error, shape of raw image and labeled image do not match')
		return None


	### FOREGROUND  VARIATIONS #####################################
	

	
	single_image_foreground = single_image 
	single_image_raw_foreground = single_raw_image 
	
	
	# Apply transform to image data
	
	
	single_raw_image_foreground = foreground_seq.augment_image(single_raw_image)
	single_image_foreground = foreground_seq.augment_image(single_image)
	# single_image_foreground= single_image_foreground*255

	
	
	black_pixels = (single_image_foreground[:,:,0] <=0.4 ) & ( single_image_foreground[:,:,1] <= 0.4) & (single_image_foreground[:,:,2] <= 0.4 )
	black_pixels = np.where(black_pixels==True)
	replacement_values  = np.average (np.average(single_raw_image, axis =0 ), axis = 0) 
	# print('replacement_values', replacement_values)
	
	single_image_foreground[black_pixels[0],black_pixels[1],: ] = replacement_values 
	single_raw_image_foreground[black_pixels[0],black_pixels[1],:  ] = replacement_values 
           	
	if(show_plots== True ): 
		f,ax = plt.subplots(2, sharey =True)
		ax[0].imshow(single_image_foreground)
		ax[1].imshow(single_raw_image_foreground)
		
		plt.show()

            ### BACKGROUND VARIATIONS #####################################
	
	single_image = single_image_foreground
	single_raw_image = single_raw_image_foreground
            
	image = single_image
            # for labeled Image get the bool values for the inside outside, boundary and exterior 
	inside_bool = (image[:,:,0] >= 150 ) & ( image[:,:,1] <= 120) & (image[:,:,2] <= 130 )
   	boundary_bool = (image[:,:,0] <=120  ) & ( image[:,:,1] >= 150) & (image[:,:,2] <= 120 )
   	exterior_bool = ~inside_bool & ~boundary_bool
    
	## inside pixels 
	inside_pixels = np.where(inside_bool==True)
	
	## boundary pixels 
	boundary_pixels = np.where(boundary_bool==True)

            ## outside pixels 
	exterior_pixels = np.where(exterior_bool==True)
	
	
	# apply augmentation to the foreground image 
	single_raw_image_background = background_seq.augment_image(single_raw_image_foreground)
	single_image_background = background_seq.augment_image(single_image_foreground)
	
	if(show_plots== True ): 
		f,ax = plt.subplots(2, sharey =True)
		ax[0].imshow(single_image_background)
		ax[1].imshow(single_raw_image_background)
		
		plt.show()



	# get the inside and boundary pixels from the normal image
	single_raw_image_background[inside_pixels[0],inside_pixels[1],:]  = single_raw_image_foreground[inside_pixels[0],inside_pixels[1],:]
	single_raw_image_background[boundary_pixels[0],boundary_pixels[1],:]  = single_raw_image_foreground[boundary_pixels[0],boundary_pixels[1],:]


	single_image_background[inside_pixels[0],inside_pixels[1],:]  = single_image_foreground[inside_pixels[0],inside_pixels[1],:]
	single_image_background[boundary_pixels[0],boundary_pixels[1],:]  = single_image_foreground[boundary_pixels[0],boundary_pixels[1],:]

	


	return(single_raw_image_background, single_image_background)

###################END OF FUNCTION#########################

folder = 'labeled/'
raw_folder = 'raw-rgb/'
files  = os.listdir(folder)
raw_files = os.listdir(raw_folder)

# Open images 
images = [cv2.imread(folder+x) for x in files]
raw_images = [cv2.imread(raw_folder+x) for x in raw_files]


for im in range(0,len(images)): 

	single_image = images[0]
	single_raw_image = raw_images[0]

	print(single_raw_image.shape)


	# foreground transform
	# Create Afine transform

	shear_value = random.randrange(-25,25)
	dropout_range = random.uniform(0.01,0.3)
	gaussian_range = random.randrange(3.0,10.0)
	foreground_seq = iaa.Sequential([ iaa.Affine( shear=( shear_value))]) # shear by -16 to +16 degrees

	# background transform 
	background_seq = iaa.Sequential([
		   iaa.Dropout((0.0, dropout_range), per_channel=0.5),
		  # iaa.Add((-0.2, 0.2), per_channel=0.5),
		    iaa.GaussianBlur(sigma=(2.00, gaussian_range)) # blur images with a sigma of 0 to 3.0
		])


	single_raw_image_background, single_image_background = augment_images(foreground_seq, background_seq, single_image, single_raw_image, show_plots =False)
	cv2.imwrite(files[im]+'_augmented.png',  single_image_background)
	cv2.imwrite( raw_files[im]+'_augmented.png', single_raw_image_background)
	
	# new_image = foreground_seq.augment_image(single_image)
	# plt.imshow(new_image)
	# plt.show()

#################################################################
#################################################################
#################################################################







	# # for labeled Image
	# inside_bool = (single_image[:,:,0] >= 170 ) & ( single_image[:,:,1] <= 120) & (single_image[:,:,2] <= 130 )
	# boundary_bool = (single_image[:,:,0] <=120  ) & ( single_image[:,:,1] >= 150) & (single_image[:,:,2] <= 120 )
	# exterior_bool = ~inside_bool & ~boundary_bool

	# ## inside pixels 
	# inside_pixels = np.where(inside_bool==True)
	
	# ## boundary pixels 
	# boundary_pixels = np.where(boundary_bool==True)
	
	# all_pixels = inside_pixels +boundary_pixels






# black_pixels  = np.where((single_image_foreground[:,:,0] == 0 ) & (single_image_foreground[:,:,1] == 0 ) &(single_image_foreground[:,:,2] == 0 ))  
	# black_pixels_list = np.where(black_pixels == True)
	# average_pixels = np.average(np.average(single_raw_image, axis=0 ), axis=0)
	
	# single_raw_image2[inside_pixels[0],inside_pixels[1],:] = single_raw_image_foreground[inside_pixels[0],inside_pixels[1],:] 
	# single_raw_image2[boundary_pixels[0],boundary_pixels[1],:] = single_raw_image_foreground[boundary_pixels[0],boundary_pixels[1],:] 

	# single_image2[inside_pixels[0],inside_pixels[1],:] = single_image_foreground[inside_pixels[0],inside_pixels[1],:] 
	# single_image2[boundary_pixels[0],boundary_pixels[1],:] = single_image_foreground[boundary_pixels[0],boundary_pixels[1],:] 

	
	# if(show_plots== True ): 
	# 	f,ax = plt.subplots(2,2, sharex =True)
	# 	ax[0,0].imshow(single_raw_image2)
	# 	ax[0,1].imshow(single_raw_image)
	# 	ax[1,0].imshow(single_image2)
	# 	ax[1,1].imshow(single_image)

	# 	plt.show()

	# single_raw_image2[exterior_pixels[0],exterior_pixels[1],:] = single_raw_image_background[exterior_pixels[0],exterior_pixels[1],:] 

	
	# single_image2[exterior_pixels[0],exterior_pixels[1],:] = single_image_background[exterior_pixels[0],exterior_pixels[1],:] 

	# print(single_raw_image2.shape)
	# single_image2[inside_pixels[0],inside_pixels[1] ]  

	# print(inside_pixels[0])
	# plt.imshow(inside_bool, cmap=plt.cm.gray, interpolation='nearest')
	# plt.show()
	# if(show_plots== True ):
	# 	f,ax = plt.subplots(2,2,sharex =True)
	# 	ax[0,0].imshow(single_raw_image2)
	# 	ax[0,1].imshow(single_raw_image)
	# 	ax[1,0].imshow(single_image2)
	# 	ax[1,1].imshow(single_image)

	# 	plt.show()


