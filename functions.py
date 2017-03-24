import cv2
import matplotlib.pyplot as plt
from skimage.io import imread
from keras import backend as K
import numpy as np

def resize_crop_image(image,scale,cutoff_percent):
	image = cv2.resize(image,None,fx=scale, fy=scale, interpolation = cv2.INTER_AREA)
	cut_off_vals = [image.shape[0]*cutoff_percent/100, image.shape[1]*cutoff_percent/100]


	end_vals = [image.shape[0]-int(cut_off_vals[0]),image.shape[1]-int(cut_off_vals[1])]

	image =image[int(cut_off_vals[0]):int(end_vals[0]),int(cut_off_vals[1]):int(end_vals[1])  ]
	#plt.imshow(image)
	#plt.show()
	return(image)

def rotate_thrice(square):
        return [square, np.rot90(square, 1), np.rot90(square, 2), np.rot90(square, 3)]

def transforms(square):
        return rotate_thrice(square) + rotate_thrice(np.fliplr(square))

def your_loss(y_true, y_pred):
	#weights = np.ones(4)
	#weights = np.array([ 4.2 ,  0.52,  1.3,  0.08])
        #weights = np.array([0.99524712791495196, 0.98911715534979427, 0.015705375514403319])
        weights = np.array([ 0.91640706, 0.5022308, 0.1])
	#weights = np.array([ 0.05 ,  1.3,  0.55,  4.2])
	#weights = np.array([0.00713773, 0.20517703, 0.15813273, 0.62955252])
	#weights = np.array([1,,0.1,0.001])
	# scale preds so that the class probas of each sample sum to 1
	y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
	# clip
	y_pred = K.clip(y_pred, K.epsilon(), 1)
	# calc
	loss = y_true*K.log(y_pred)*weights
	loss =-K.sum(loss,-1)
	return loss

def raw_to_labels(image):
    assert(image.max()==255)
    inside_bool = (image[:,:,0] <= 120 ) & ( image[:,:,1] <= 120) & (image[:,:,2] >= 130 )
    boundary_bool = (image[:,:,0] <=120  ) & ( image[:,:,1] >= 150) & (image[:,:,2] <= 120 )
    exterior_bool = ~inside_bool & ~boundary_bool

    softmax_labeled_image = np.zeros(image.shape)
    softmax_labeled_image[inside_bool] = [1,0,0]
    softmax_labeled_image[boundary_bool] = [0,1,0]
    softmax_labeled_image[exterior_bool] = [0,0,1]
    return softmax_labeled_image
