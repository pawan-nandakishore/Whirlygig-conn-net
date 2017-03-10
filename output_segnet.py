from keras.models import load_model
import matplotlib.pyplot as plt
from skimage import io
from skimage.io import imread
import numpy as np
from skimage.color import gray2rgb
from skimage.io import imsave
from datetime import datetime
from keras import backend as K
from conv_segnet import your_loss

#img = io.imread('images/fucked.png')
#gray = io.imread('images/fucked.png', as_grey=True)

model = load_model('auto.h5', custom_objects={'your_loss': your_loss})

img = imread('images/raw_image_cropped.png', as_grey=True)
#labels = np.load('labels.npy')
#labels_280 = np.zeros((280,280,4))
#labels_280[:-1,:-1,:]=labels
#print(labels_280.shape)

# Add the extra row
grey = np.zeros((280,280))
grey[:-1,:-1] = img

xs = grey.reshape(1,1,280,280)

result = model.predict(xs).reshape(280,280,4)

#count = 0

zeros = np.zeros((280,280,4))

for i in range(grey.shape[0]):
    for j in range(grey.shape[1]):
        output = result[i,j]
        zeros[i,j,np.argmax(output)] = 255
        #count += 1

zeros[:,:,3]=255

#print(count-len(squares2))
plt.imshow(zeros/255)
#plt.imsave('plots/new.png', zeros)
plt.show()
#
