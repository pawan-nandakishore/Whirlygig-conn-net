import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
#img = imread('6.jpg')
#label = imread('6.jpg')
img = imread('0.png')
label = imread('lab.png')

#img = np.zeros((40,40))

#img[10,:]=1

#plt.imshow(img)

overlap = 10

images = [img]*10
labels = [label]*10#[img[overlap:img.shape[0]-overlap,overlap:img.shape[1]-overlap]]*100



st = lambda aug: iaa.Sometimes(1, aug)

seq = iaa.Sequential([
        #st(iaa.Multiply((0.5, 1.5), per_channel=0.5))
        #st(iaa.Add((10, 100)))
        #st(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(2,5)))
        st(iaa.ElasticTransformation(alpha=12, sigma=3)),
        #st(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)),
        #st(iaa.Invert(0.25, per_channel=True)),
        #st(iaa.Dropout((0.0, 0.1), per_channel=0.5)),
        #st(iaa.GaussianBlur((0, 1))),
        #st(iaa.AdditiveGaussianNoise(loc=0, scale=(0, 50), per_channel=0.5))
        #st(iaa.Affine(
            #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            #translate_px={"x": (-16, 16), "y": (-16, 16)}, # translate by -16 to +16 pixels (per axis)
        #    rotate=(-90, 90), # rotate by -45 to +45 degrees
            #shear=(-16, 16), # shear by -16 to +16 degrees
            #order=ia.ALL, # use any of scikit-image's interpolation methods
            #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
        #    mode="reflect" # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        #))
    ], random_order=True)

seq_det = seq.to_deterministic()
images_aug = seq_det.augment_images(images)
heat_aug = seq_det.augment_images(labels)

for i,img in enumerate(images_aug):
    #im2 = img[overlap:img.shape[0]-overlap,overlap:img.shape[1]-overlap]*0.5 + heat_aug[i]*0.5
    img2 = heat_aug[i]*0.5 + img[overlap:img.shape[0]-overlap,overlap:img.shape[1]-overlap]*0.5
    img[overlap:img.shape[0]-overlap,overlap:img.shape[1]-overlap] = img2
    #print(img)
    imsave('elastic/%d.png'%i, img)
    plt.figure()
    plt.imshow(img)
    #imsave('elastic/%d.png'%i, heat_aug[i])
#plt.imshow(img)
#plt.figure()
#plt.imshow(images_aug[0])
#plt.show()
