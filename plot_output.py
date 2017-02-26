from skimage.io import imread
import matplotlib.pyplot as plt

img = imread('fig.tiff')
plt.imshow(img)
plt.show()


