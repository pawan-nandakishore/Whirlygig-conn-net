#import numpy as np
from scipy.io import loadmat
from skimage.io import imread, imsave
import random
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import gray2rgb

# Also try unit test with 5 n 3 linear convolution
img = imread('raw_image.png', as_grey=True)
output = np.load('red_pixel_class.npy')

img = img[400:1000,400:920]
output = output[400:1000,400:920]


# Test that it still works
#plt.imshow(rgb)
#plt.show()

M, N = img.shape
c = 31
sample_size = 1486

junctions = []
others = []
squares = []
types = []

def generateSamples(data, classi, num):
    samples = random.sample(data, num)
    for i, sample in enumerate(samples):
        plt.imshow(sample)
        filename = ('%s/%d.png')%(classi, i)
        print(filename)
        plt.savefig(filename)
        plt.clf()

# Run the square over the entire image
for i in range(c/2, M-c/2):
    for j in range(c/2, N-c/2):
        square = img[i-c/2:i+c/2+1, j-c/2:j+c/2+1]
        if output[i][j] == 255:
            junctions.append(square)
        else:
            others.append(square)


others = random.sample(others, sample_size)
print(len(others), len(junctions))
# Equally sample from both
squares = junctions + others
ys = [1 for i in xrange(sample_size)] + [0 for i in xrange(sample_size)]
#samples = random.sample(data, num)
xs = np.array(squares)
ys = np.array(ys)
print(xs.max())
np.save('xs.npy', xs)
np.save('ys.npy', ys)
#generateSamples(junctions, 'junctions', sample_size)
#generateSamples(others, 'others', sample_size)
#generateSamples(exteriors, 'exteriors', sample_size)
