import numpy as np
from skimage.io import imread, imsave
from skimage.transform import rotate
import random
import matplotlib.pyplot as plt
from skimage.color import rgb2grey
import time

images = ['images/raw_image_cropped2.png']
labels = np.array([np.load('labels.npy')]) # This needs scaling
print(labels.shape)

def rotate_thrice(square):
	return [square, rotate(square, 90), rotate(square, 180), rotate(square, 270)]

def transforms(square):
	return rotate_thrice(square) + rotate_thrice(np.fliplr(square))

for idx, fl in enumerate(images):
	img = imread(fl, as_grey=True) #Grayscale images are already scaled between 0 and 1
	#rz = imread(fl) #Grayscale images are already scaled between 0 and 1
	label = labels[idx]
	#rz[label==1] = [255,0,0,255]

	# Do the crop
	M, N = img.shape
	c = 17
	junctions = []
	exteriors = []
	boundaries = []
	interiors = []

	datas = [junctions, boundaries, interiors, exteriors]

	for i in range(c/2, M-c/2):
		for j in range(c/2, N-c/2):
			#print(i,j)
			square = img[i-c/2:i+c/2+1, j-c/2:j+c/2+1]
			# sq_rz = rz[i-c/2:i+c/2+1, j-c/2:j+c/2+1, :].copy()
			# if sq_rz[10,10][0] == 255:
                                                        				# sq_rz[10,10]=[0,0,255,255]
			# else:
				# sq_rz[10,10]=[0,255,0,255]

			idx = np.argmax(label[i,j])

			squares = transforms(square)

			datas[idx].extend(squares)

			 #squares.extend()

			#if label[i][j][0] ==
				 #squares_rez = transforms((square))

				 #for y in xrange(3):
				 #for x in xrange(4):
				#sq=square.copy()
				#sq[c/2,c/2]=1
				#imsave(('images/vid.png'), sq)
				#time.sleep(0.25)

			#	junctions.extend(squares)
			#else:
			#	others.extend(squares)

print(len(junctions), len(boundaries), len(interiors), len(exteriors))

# Sample (1075, 5603, 5429, 57062)WWW
junctions = random.sample(junctions, 8600)
boundaries = random.sample(boundaries, 5000)
interiors = random.sample(interiors, 5000)
exteriors = random.sample(exteriors, 3000)

squares = junctions + boundaries + interiors + exteriors
ys = [[1,0,0,0]]*len(junctions)+[[0,1,0,0]]*len(boundaries) + [[0,0,1,0]]*len(interiors) + [[0,0,0,1]]*len(exteriors)

#sample_size = len(junctions)
#others_sample_size = 8*len(junctions)
#others = random.sample(others, others_sample_size)
#print(len(others), len(junctions))
## Equally sample from both
#squares = junctions + others
#ys = [1 for i in xrange(sample_size)] + [0 for i in xrange(others_sample_size)]
##samples = random.sample(data, num)
xs = np.array(squares)
ys = np.array(ys)
#
assert(len(xs)==len(ys))
assert((1-xs.max()<0.1) and xs.min()==0)
#
np.save('data/xs.npy', xs)
np.save('data/ys.npy', ys)
#
#
