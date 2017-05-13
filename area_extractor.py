import os
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import glob
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.patches as mpatches
from skimage.color import label2rgb
import random


def calculate_stats(data):
    mean = np.mean(data)
    std = np.std(data)
    confidence = 3*std
    left = mean - confidence
    right = mean + confidence
    return mean, std, left, right

def lies_outside(x, mean, std, left, right):
    return x<left or x>right

def get_blue_region(image):
    inside_bool = (image[:,:,0] <= 120 ) & ( image[:,:,1] <= 120 ) & (image[:,:,2] >= 130 )
    inside_bool = inside_bool.astype(int)
    return inside_bool

def get_regions(image):
    image_bw = get_blue_region(image)
    label_image = label(image_bw)
    regions = regionprops(label_image)
    return regions
    #regions_g.extend(region)

if __name__ == "__main__":
    output_files = sorted(glob.glob('plots/results/*'))
    raw_files =  ['images/cropped/rgbs/'+os.path.basename(fl) for fl in output_files]
    print("Output files: %s, raw files: %s" % (output_files, raw_files))

    raw_images = []
    output_images = []

    attributes_g = []
    regions_g = []

    square_size = 38
    box_dim_x = 56
    box_dim_y = 36

    # Read all regions
    for i,fl in enumerate(output_files):
        print("Reading: %s, %d/%d"% (fl,i,len(output_files)))
        image_out = imread(fl)
        image_in = imread(raw_files[i]) # Possible paddding needed here. Would get an error otherwise
        regions = get_regions(image_out)

        for r in regions:
            r.input = image_in
            r.output = image_out
            regions_g.append(r)

    # Histogram of solidity
    attributes_g = [getattr(r, 'solidity') for r in regions_g]
    mean, std, left, right = calculate_stats(attributes_g)
    #plt.hist(attributes_g, bins=20)
    #plt.show()

    # Caclulate bounding box stats
    box_h, box_w = zip(*[(r.bbox[2]-r.bbox[0], r.bbox[3]-r.bbox[1]) for r in regions_g])
    box_h_mean, box_h_std, _,_ = calculate_stats(box_h)
    box_w_mean, box_w_std, _,_ = calculate_stats(box_w)
    box_h_conf = box_h_mean + 1.5*box_h_std
    box_w_conf = box_w_mean + 1.5*box_w_std
    #box_dim = int(max(box_h_conf, box_w_conf))
    #diff = (square_size - box_dim)/2
    #print(box_h_mean, box_w_mean, box_h_conf, box_w_conf, box_dim)
    #plt.hist(box_h, bins=30)
    #plt.hist(box_w, bins=30)
    #plt.imshow(regions_g[0].output)
    #plt.show()
    images_g = []
    images_o = []

    huge = np.ones((1080, 1080))*255
    huge_out = np.ones((1080, 1080, 4))*255

    # Find outlier
    for r in regions_g:
        if lies_outside(r.solidity, mean, std, left, right):
            row, col = r.centroid
            row = int(row)
            col = int(col)
            #print(row, col, box_dim/2)
            img_cropped = r.input[row-box_dim_x/2:row+box_dim_x/2,col-box_dim_x/2:col+box_dim_x/2]
            img_cropped_o = r.output[row-box_dim_y/2:row+box_dim_y/2,col-box_dim_y/2:col+box_dim_y/2]
            
            if img_cropped.shape[0] == img_cropped.shape[1]:
                images_g.append(img_cropped.copy())
                images_o.append(img_cropped_o.copy())
            #plt.imshow(img_cropped)
            #plt.show()


    data = zip(images_g, images_o)
    errors_length = len(data)
    #data = random.sample(data, min(len(images_g),num_squares**2))
    # Sample the top 400 contours, assume that you have more
    data = random.sample(data, min(len(images_g),64))
    images_g, images_o = zip(*data)
    #plt.imshow(images_g[0])
    #images_g = random.sample(images_g, min(len(images_g),num_squares**2))
    print("Erroneous contours: %d/%d"%(errors_length, 200*len(output_files)), images_g[0].shape)
    
    #for i,(img_g, img_o) in enumerate(zip(images_g, images_o)):
    #    print('plots/comparisons/errors_i/%d.png'%i, img_g.shape)
    #    imsave('plots/comparisons/errors_i/%d.png'%i, img_g)
    [plt.imsave('plots/comparisons/errors_i/%04d.png'%i, img) for i, img in enumerate(images_g)]
    [plt.imsave('plots/comparisons/errors_o/%04d.png'%i, img) for i, img in enumerate(images_o)]
