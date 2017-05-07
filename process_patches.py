import numpy as np
import glob
from scipy.misc import imread
import random
from functions import sort_by_number, raw_to_labels, plot_row
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import matplotlib.pyplot as plt

def zip_and_sample(a,b,n):
    """ Takes two lists of objects a,b and samples n from both while preserving correspondence
    
    Args:
        a(list): First list
        b(list): Second list
        n(int): How many items to sample
        
    Returns:
        x(list): n samples of zippped items, can we return them separately
    
    """
    data = zip(a, b)
    data_sampled = random.sample(data, n)
    x, y = zip(*data_sampled)
    return list(x), list(y)
    

def read_data(x_path, y_path, num):
    """ Reads input images from x_path and their labels from y_path and returns a sampled subset
    
    Args:
        x_path(list(str)): List of paths of input x
        y_path(list(str)): List of paths of input y
        num(int): How many images to sample
        
    Returns:
        x_tensor(tensor): Tensor of rgb images
        y_tensor(tensor): Tensor of label images
    
    """
    x_names = sort_by_number(x_path)
    y_names = sort_by_number(y_path)
    
    x_names_sampled, y_names_sampled = zip_and_sample(x_names, y_names, num)
    
    x_tensor = load_tensor(x_names_sampled)
    y_tensor = load_tensor(y_names_sampled)
    
    return x_tensor, y_tensor
    
def load_tensor(path):
    """ Loads a list of entities from path, applied lambdaFunc to them and returns the resulting tensor
    
    Arguments:
        path(list(str)): Path to read list of images from
        
    Returns:
        itemTensor(tensor): Tensor of all objects with lambdaFunc applied to each of them
    
    """
    imgs = [imread(fl, mode='RGB') for fl in path]
    return np.array(imgs).astype(float)

def augment_tensor(x_tensor, y_tensor):
    """ Performs on the fly augmentation on a batch of x, y values and returns augmented tensor 
    
    Args:
        x_tensor(tensor): Tensor of x images
        y_tensor(tensor): Tensor of y images
        
    Returns:
        x_aug_tensor(tensor): Tensor of augmented x images
        y_aug_tensor(tensor): Tensor of augmented y images
   
    """
    st = lambda aug: iaa.Sometimes(1, aug)
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
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
		#    rotate=(-45, 45), # rotate by -45 to +45 degrees
		    #shear=(-16, 16), # shear by -16 to +16 degrees
		    #order=ia.ALL, # use any of scikit-image's interpolation methods
		    #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
		    #mode="reflect" # use any of scikit-image's warping modes (see 2nd image from the top for examples)
		#))
	    ], random_order=True)

    seq_det = seq.to_deterministic()
    x_aug_tensor = seq_det.augment_images(x_tensor)
    y_aug_tensor = seq_det.augment_images(y_tensor)
    return x_aug_tensor, y_aug_tensor

def fetch_batch(xpath, ypath, n):
    """ Fetches batch of size n. Add tests for this method. Really important that this is not wrong """
    x, y = read_data(glob.glob(xpath), glob.glob(ypath), n)
    #x_aug, y_aug = augment_tensor(x, y)
    x_aug, y_aug = x, y

    x_aug = x_aug/255
    y_aug = np.array([raw_to_labels(y) for y in y_aug])
    
    return x_aug, y_aug
        
def yield_batch(xpath, ypath, n=64):
    """ Yields batch of size n infinitely """
    while True:
        x_aug, y_aug = fetch_batch(n)
        yield (x_aug, y_aug)
    
    #plt.imshow(y_aug[0])
    
def visualize_batch(x,y):
    """ Visualize batch of x, y examples """
    for idx, (i,j) in enumerate(zip(x,y)):
        img = i
        overlap = 10
        img[overlap:img.shape[0]-overlap,overlap:img.shape[1]-overlap,:] = j
        #plt.imsave('outs/%d_i.png'%idx, i)
        plt.imsave('outs/%d.png'%idx, img)
    

#if __name__ == "__main__":
#x, y = gen_batch(64)


    #plt.imshow(j)
    #plot_row('new.py', x,y)
    #plt.show()
