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
    confidence = 0*std
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
    # Read files
    #output_files = sorted(glob.glob('comparison/joint+u+weighted/*.png'))
    #output_files = sorted(glob.glob('/home/pavan/Dropbox/Whirlygig/*.png'))
    #print(output_files)
    output_files = sorted(glob.glob('/home/monisha/Downloads/results_elastic/*.png'))
    #raw_files = sorted(glob.glob('cleaned/raw/*.png'))
    #output_files = random.sample(output_files, 300)
    raw_files =  ['images/rgbs/'+os.path.basename(fl).replace('.png','')+'.png' for fl in output_files]
    #raw_files =  ['images/single_frames/'+fl.split('/')[-1] for fl in output_files]
    #raw_files =  ['cleaned/all/'+'_'.join(fl.split('_')[1:]) for fl in output_files]
    print("Output files: %s, raw files: %s" % (output_files, raw_files))

    raw_images = []
    output_images = []

    attributes_g = []
    regions_g = []

    square_size = 36

    # Read all regions
    for i,fl in enumerate(output_files):
        print("Reading: %s, %d/%d"% (fl,i,len(output_files)))
        image_out = imread(fl)
        image_in = imread(raw_files[i])
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
    box_dim = 34
    diff = (square_size - box_dim)/2
    print(box_h_mean, box_w_mean, box_h_conf, box_w_conf, box_dim)
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
            print(row, col, box_dim/2)
            img_cropped = r.input[row-box_dim/2:row+box_dim/2,col-box_dim/2:col+box_dim/2]
            img_cropped_o = r.output[row-box_dim/2:row+box_dim/2,col-box_dim/2:col+box_dim/2]
            images_g.append(img_cropped.copy())
            images_o.append(img_cropped_o.copy())
            #plt.imshow(img_cropped)
            #plt.show()


    num_squares = 1080/square_size
    data = zip(images_g, images_o)
    errors_length = len(data)
    #data = random.sample(data, min(len(images_g),num_squares**2))
    # Sample the top 400 contours, assume that you have more
    data = random.sample(data, min(len(images_g),400))
    images_g, images_o = zip(*data)
    #images_g = random.sample(images_g, min(len(images_g),num_squares**2))
    print("Erroneous contours: %d/%d"%(errors_length, 200*len(output_files)))
    [imsave('plots/comparisons/errors_i/%d.png'%i, img) for i, img in enumerate(images_g)]
    [imsave('plots/comparisons/errors_o/%d.png'%i, img) for i, img in enumerate(images_o)]
    #print("Num squares: %d", num_squares)

    #y = len(images_g)/num_squares
    #print(y)
    #x = len(images_g)%num_squares

    #if y==0:
    #    y=1

    #print(len(images_g))
    #for i in xrange(y):
    #    for j in xrange(num_squares):
            #print(images_g[i*16+j].shape)
            #print(square_size*i+diff,square_size*(i+1)-diff,square_size*j+diff,square_size*(j+1)-diff)
    #        huge[square_size*i+diff:square_size*(i+1)-diff,square_size*j+diff:square_size*(j+1)-diff] = images_g[i*box_dim+j]
    #        huge_out[square_size*i+diff:square_size*(i+1)-diff,square_size*j+diff:square_size*(j+1)-diff] = images_o[i*box_dim+j]
            #huge[i+2:i+square_size,j+2:j+square_size,:] = images_g[i*16+j]
    #plt.imshow(huge)
    #plt.show()

    #imsave('output_i.png', huge/255)
    #imsave('output_o.png', huge_out/255)
    #plt.imshow(huge/255, cmap='Greys')
    #plt.show()

    # Find outliers
    """for fl in output_files:
        fig, ax = plt.subplots(figsize=(10, 6))
        image = imread(fl)
        print("Reading: %s"% fl)
        image_bw = get_blue_region(image)
        #con_hull = convex_hull_object(image_bw)
        label_image = label(image_bw)
        regions = regionprops(label_image)
        #plt.imshow(label_image)
        #plt.show()
        #print(image.shape, label_image.shape)
        #image_label_overlay = label2rgb(label_image, image)
        ax.imshow(image_bw, cmap='Greys')
        #plt.show()
        #regions, attributes = get_regions(image_bw)

        for r in regions:
            if lies_outside(r.solidity, mean, std, left, right):
                minr, minc, maxr, maxc = r.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
                img2 = r.image
                plt.imshow(img2, cmap='Greys')
                plt.show()

        ax.set_axis_off()
        plt.tight_layout()
        plt.show()

    #ax.set_axis_off()
    #plt.tight_layout()
    #plt.show()
    """
    # print(mean, std, left, right)
