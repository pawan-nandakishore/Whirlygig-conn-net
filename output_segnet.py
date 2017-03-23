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
import glob
from scipy.misc import imread

#img = io.imread('images/fucked.png')
#gray = io.imread('images/fucked.png', as_grey=True)

model = load_model('weights.hdf5', custom_objects={'your_loss': your_loss})

files = glob.glob('cropped/*')
for fl in files:
	print("Processing: %s"%fl)
	img = imread(fl, mode='RGB')
	#img = imread('images/raw_image_cropped.png', as_grey=True)
	#labels = np.load('labels.npy')
	#labels_432 = np.zeros((432,432,4))
	#labels_432[:-1,:-1,:]=labels
	#print(labels_432.shape)

	# Add the extra row
	#grey = np.zeros((432,432))
	#grey[:-1,:-1] = img

	xs = img.reshape(1,3,432,432)

	result = model.predict(xs).reshape(432,432,3)

	#count = 0

	zeros = np.zeros((432,432,4))

	for i in range(img.shape[0]):
	    for j in range(img.shape[1]):
		output = result[i,j]
		zeros[i,j,np.argmax(output)] = 255
		#count += 1

	zeros[:,:,3]=255

	#print(count-len(squares2))
	plt.imshow(zeros/255)
	plt.imsave('plots/%s'%fl, zeros/255)
	#plt.show()
