import cv2
import matplotlib.pyplot as plt
from skimage.io import imread

def resize_crop_image(image,scale,cutoff_percent):
    image = cv2.resize(image,None,fx=scale, fy=scale, interpolation = cv2.INTER_AREA)
    cut_off_vals = [image.shape[0]*cutoff_percent/100, image.shape[1]*cutoff_percent/100]


    end_vals = [image.shape[0]-int(cut_off_vals[0]),image.shape[1]-int(cut_off_vals[1])]

    image =image[int(cut_off_vals[0]):int(end_vals[0]),int(cut_off_vals[1]):int(end_vals[1])  ]
    #plt.imshow(image)
    #plt.show()
    return(image)

#img = imread('marked_image2_3cl')
#resize_crop_image(img, 0.25, 15)
