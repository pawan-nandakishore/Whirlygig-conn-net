from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave
from scipy.misc import imread
from skimage.color import gray2rgb
import glob
from functions import sort_by_number, squares_to_tiles
from tqdm import tqdm

def c_weight(image):
            junctions_bool = (image[:,:,0]>=150) & ( image[:,:,1] <= 120) & (image[:,:,2] <= 120 )
            boundary_bool = (image[:,:,0] <=120  ) & ( image[:,:,1] >= 150) & (image[:,:,2] <= 120 )
            inside_bool = (image[:,:,0] <= 120 ) & ( image[:,:,1] <= 120) & (image[:,:,2] >= 130 )
            exterior_bool = ~inside_bool & ~boundary_bool & ~junctions_bool
            
            exterior_count = image[exterior_bool].shape[0]
            other_count = image[~exterior_bool].shape[0]
            
            return ((other_count+0.0001)/(exterior_count+other_count)>0.3)
        
#def convolution_sampling
        
def sample_squares_hyperstack(img_path, labels_path):
    """ Sample fixed no of squares from hyperstack of image data """
    tiles = []
    
    
    for img_fl, label_fl in tqdm(zip(img_path, labels_path)):
        img = imread(img_fl, mode='RGB')
        tiles.extend(squares_to_tiles(img, (36,36), (1,1)))
        
    tiles = np.array(tiles)
    weights = np.apply_along_axis(c_weight, tiles)
    print(weights.shape)
    #print(len(tiles), len(indices))
        #labels.extend(square)
        

if __name__ == "__main__":

    img_path = sort_by_number(glob.glob('images/rgbs/*'))
    labels_path = sort_by_number(glob.glob('images/labeled/*.png'))
    
    sample_squares_hyperstack(img_path, labels_path)
    
    
    
#[imsave("images/patches/combined/%d.png"%idx, img) for idx, img in enumerate(images)]
#[imsave("images/patches/xs/%d.png"%idx, img) for idx, img in enumerate(xs)]
#[imsave("images/patches/ys/%d.png"%idx, img) for idx, img in enumerate(ys)]

