from skimage.io import imread, imsave
import matplotlib.pyplot as plt

img = imread('path1.png')
img_labeled = imread('path2.png')

def crop(img, row, col, row_s, col_s):
    return img[row:row+row_s,col:col+col_s]

regions = [(221,251),(814,193),(601,365),(422,426),(798,472),(526,527),(347,576),(510,728),(793,725),(532,851)]

#
images = [crop(img, r[1], r[0],26,26) for r in regions]
images_labeled = [crop(img_labeled, r[1], r[0],26,26) for r in regions]

[imsave(('cropped/raw/%d.png'%i), images[i]) for i in xrange(len(images))]
[imsave(('cropped/labeled/%d.png'%i), images_labeled[i]) for i in xrange(len(images))]
