from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import glob
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.patches as mpatches
from skimage.color import label2rgb

def calculate_stats(data):
    mean = np.mean(data)
    std = np.std(data)
    confidence = 1.5*std
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
    areas = [r.solidity for r in regions]
    return regions, areas
    #regions_g.extend(region)

results = glob.glob('plots/results/*')

areas_g = []

for fl in results:
    image = imread(fl)
    regions, areas = get_regions(image)
    areas_g.extend(areas)

areas_g = np.array(areas_g)
areas_g.shape
mean, std, left, right = calculate_stats(areas_g)
plt.hist(areas_g, bins=20)
plt.show()

for fl in results:
    fig, ax = plt.subplots(figsize=(10, 6))
    image = imread(fl)
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
    #regions, areas = get_regions(image_bw)

    for r in regions:
       if lies_outside(r.solidity, mean, std, left, right):
           minr, minc, maxr, maxc = r.bbox
           rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
           ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

#ax.set_axis_off()
#plt.tight_layout()
#plt.show()

# print(mean, std, left, right)


