import numpy as np
from skimage.io import imread, imsave
import random

img = imread('images/raw_image_cropped.png', as_grey=True) #Grayscale images are already scaled between 0 and 1
labels = np.load('data/labeled_image.npy')/255 # This needs scaling

# Do the crop
M, N = img.shape
c = 6

junctions = []
others = []

for i in range(c/2, M-c/2):
    for j in range(c/2, N-c/2):
        square = img[i-c/2:i+c/2+1, j-c/2:j+c/2+1]
        if labels[i][j] == 1:
            junctions.append(square)
        else:
            others.append(square)

sample_size = len(junctions)
others_sample_size = 4*len(junctions)
others = random.sample(others, others_sample_size)
print(len(others), len(junctions))
# Equally sample from both
squares = junctions + others
ys = [1 for i in xrange(sample_size)] + [0 for i in xrange(others_sample_size)]
#samples = random.sample(data, num)
xs = np.array(squares)
ys = np.array(ys)

assert(len(xs)==len(ys))
assert(xs.max()==1 and xs.min()==0)

np.save('data/xs.npy', xs)
np.save('data/ys.npy', ys)


