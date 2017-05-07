from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave
from scipy.misc import imread
from skimage.color import gray2rgb
import glob
from functions import sort_by_number

img_files = sort_by_number(glob.glob('images/rgbs/*'))
label_files = sort_by_number(glob.glob('images/labeled/*.png'))

print(img_files, label_files)

images = []
xs = []
ys = []
weights = []

def c_weight(image):
        junctions_bool = (image[:,:,0]>=150) & ( image[:,:,1] <= 120) & (image[:,:,2] <= 120 )
        boundary_bool = (image[:,:,0] <=120  ) & ( image[:,:,1] >= 150) & (image[:,:,2] <= 120 )
        inside_bool = (image[:,:,0] <= 120 ) & ( image[:,:,1] <= 120) & (image[:,:,2] >= 130 )
        exterior_bool = ~inside_bool & ~boundary_bool & ~junctions_bool
        
        exterior_count = image[exterior_bool].shape[0]
        other_count = image[~exterior_bool].shape[0]
        
        return (other_count+0.0001)/(exterior_count+other_count)

for img_fl, lbl_fl in zip(img_files, label_files):
    img = imread(img_fl, mode='RGB')
    #img = imread(img_fl, mode='L')
    label = imread(lbl_fl, mode='RGB')
    print('Reading: %s'%img_fl)
    #img = np.random.rand(3,3)
    #plt.imshow(img)
    #plt.show()
    
    inner_size = 36
    overlap = 10
    larger_size = inner_size + overlap
    
    #plt.imshow(img_padded)
    
    img_padded = np.pad(img, ((overlap,overlap), (overlap,overlap), (0,0)), mode='reflect')
    #img_padded = gray2rgb(np.pad(img, ((overlap,overlap), (overlap,overlap)), mode='reflect'))

    
    
        
        #bools = [junctions_bool, boundary_bool, inside_bool, exterior_bool]
        #counts = [bo.shape[0] for bo in bools]
        #idx = np.argmax(counts)
        #return idx, counts[idx]
    
    # Sample based on colors
    
    # Randomly sampling
    
    count = 0
    # Get overlapping patches
    #for i in xrange(0, img.shape[0], inner_size):
    #    for j in xrange(0, img.shape[1], inner_size):
    while(count<2000):
        
            i = np.random.randint(0, img.shape[0]-inner_size)
            j = np.random.randint(0, img.shape[1]-inner_size)
            
            #print(i-overlap+overlap,i+inner_size+overlap+overlap,j-overlap+overlap, j+inner_size+overlap+overlap)
            img_overlapped = img_padded[i:i+inner_size+overlap+overlap,j:j+inner_size+overlap+overlap]
            
            colors_inside = label[i:i+inner_size,j:j+inner_size]
            weight = c_weight(colors_inside)
            
            if weight>0.3:
                img2 = img_overlapped.copy()
                img2[overlap:overlap+inner_size,overlap:overlap+inner_size] = colors_inside
                
                weights.append(weight)
                count += 1
                
                #images.append()
                images.append(img2)
                xs.append(img_overlapped)
                ys.append(colors_inside)
            
            img_inside = img[i:i+inner_size,j:j+inner_size]
            
            
            #print(weight)
            
            #imsave("generated/%d_%d.png"%(i,j), img_overlapped)
            
            #img_cropped = img[i:i+inner_size,j:j+inner_size]
            #plt.figure()
            #plt.imshow(img_cropped)
            
            #plt.figure()
            #plt.imshow(img_inside)
            
    # Get tiled patches
    
    #print(range(0, img.shape[0], inner_size))
    
    #print()
    
#plt.hist((weights), bins=20)
#plt.show()
[imsave("images/patches/combined/%d.png"%idx, img) for idx, img in enumerate(images)]
[imsave("images/patches/xs/%d.png"%idx, img) for idx, img in enumerate(xs)]
[imsave("images/patches/ys/%d.png"%idx, img) for idx, img in enumerate(ys)]
