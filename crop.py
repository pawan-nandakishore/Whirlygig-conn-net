import numpy as np
from skimage.io import imread, imsave
from skimage.transform import rotate
import random
import matplotlib.pyplot as plt
from skimage.color import rgb2grey
import time

images = ['images/raw_image_cropped.png', 'images/raw_image_cropped2.png']
labels = np.load('data/labels.npy')/255 # This needs scaling
print(labels.shape)

def rotate_thrice(square):
	return [square, rotate(square, 90), rotate(square, 180), rotate(square, 270)]

def transforms(square):
	return rotate_thrice(square) + rotate_thrice(np.fliplr(square)) + rotate_thrice(np.flipud(square))

for idx, fl in enumerate(images):
	img = imread(fl, as_grey=True) #Grayscale images are already scaled between 0 and 1
	rz = imread(fl) #Grayscale images are already scaled between 0 and 1
	label = labels[idx]
	rz[label==1] = [255,0,0,255]

	# Do the crop
	M, N = img.shape
	c = 21

	junctions = []
	others = []

	for i in range(c/2, M-c/2):
		for j in range(c/2, N-c/2):
			print(i,j)
			square = img[i-c/2:i+c/2+1, j-c/2:j+c/2+1]
			# sq_rz = rz[i-c/2:i+c/2+1, j-c/2:j+c/2+1, :].copy()
			# if sq_rz[10,10][0] == 255:
				# sq_rz[10,10]=[0,0,255,255]
			# else:
				# sq_rz[10,10]=[0,255,0,255]

			squares = transforms(square)
			if label[i][j] == 1:
				# squares_rez = transforms((sq_rz))

				# for y in xrange(3):
					# for x in xrange(4):
						# imsave(('images/vid.png'), squares_rez[y*3+x])
						# time.sleep(0.5)

				junctions.extend(squares)
			else:
				others.extend(squares)

sample_size = len(junctions)
others_sample_size = 8*len(junctions)
others = random.sample(others, others_sample_size)
print(len(others), len(junctions))
# Equally sample from both
squares = junctions + others
ys = [1 for i in xrange(sample_size)] + [0 for i in xrange(others_sample_size)]
#samples = random.sample(data, num)
xs = np.array(squares)
ys = np.array(ys)

assert(len(xs)==len(ys))
assert((1-xs.max()<0.1) and xs.min()==0)

np.save('data/xs.npy', xs)
np.save('data/ys.npy', ys)


