#import numpy as np
from scipy.io import loadmat
import random
import matplotlib.pyplot as plt

# Also try unit test with 5 n 3 linear convolution
df = loadmat('final_image_with_classes.mat')
img = df['final_image']

M, N = img.shape
c = 15
sample_size = 500

boundaries = []
interiors = []
exteriors = []

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
        if img[i][j] == 0:
            exteriors.append(square)
        elif img[i][j] == 1:
            boundaries.append(square)
        else:
            interiors.append(square)

#for i in xrange(c/2, N-c/2)


generateSamples(boundaries, 'boundaries', sample_size)
generateSamples(interiors, 'interiors', sample_size)
generateSamples(exteriors, 'exteriors', sample_size)
