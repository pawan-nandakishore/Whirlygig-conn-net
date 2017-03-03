from keras.models import load_model
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from skimage.color import gray2rgb
from skimage.io import imsave
from datetime import datetime

img = io.imread('images/raw_image_cropped.png')
output = io.imread('images/raw_image_cropped.png')
true = np.load('data/labeled_image.npy')
predicted = io.imread('plots/15_16_batch_noaug.png')

red = [255,0,0,255]

true_img = img.copy()
true_img[true==255] = red


print(img.shape)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if (predicted[i,j][0]==255) != (true[i,j]==255):
            output[i,j] = [255,0,0,255]
        if predicted[i,j][0] == true[i,j]==255:
            output[i,j] = [0,255,0,255]

fig, axes = plt.subplots(1,2,sharex='all')
axes[0].imshow(predicted)
axes[1].imshow(output)
#plt.imsave('plots/16_batch_aug.png', img)
plt.show()

