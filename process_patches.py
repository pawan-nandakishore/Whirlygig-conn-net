import numpy as np
import glob
from scipy.misc import imread
import random
from functions import sort_by_number

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
    

def sample_data(x_path, y_path, num, x_preprocess_func = lambda x: x, y_preprocess_func = lambda y: y):
    """ Reads sorted images from img_path and their labels from label_path
    
    Args:
        x_path(list(str)): List of paths of input x
        y_path(list(str)): List of paths of input y
        x_preprocess_func(function): Lambda function to apply on x, identity by default
        y_preprocess_func(function): Lambda function to apply on y, identity by default
        num(int): How many images to sample
        
    Returns:
        x_tensor(tensor): Tensor of rgb images
        y_tensor(tensor): Tensor of label images
    
    """
    x_names = sort_by_number(x_path)
    y_names = sort_by_number(y_path)
    
    x_names_sampled, y_names_sampled = zip_and_sample(x_names, y_names, num)
    
    x_tensor = load_tensor(x_names_sampled, x_preprocess_func)
    y_tensor = load_tensor(y_names_sampled, y_preprocess_func)
    
    return x_tensor, y_tensor
    
def load_tensor(path, lambdaFunc=lambda x:x):
    """ Loads a list of entities from path, applied lambdaFunc to them and returns the resulting tensor
    
    Arguments:
        path(list(str)): Path to read list of images from
        lambdaFunc(function): Any processing function to apply to these images
        
    Returns:
        itemTensor(tensor): Tensor of all objects with lambdaFunc applied to each of them
    
    """
    imgs = [lambdaFunc(imread(fl, mode='RGB')) for fl in path]
    return np.array(imgs)
    

if __name__ == "__main__":
    print(sample_data(glob.glob('images/patches/xs/*'), glob.glob('images/patches/ys/*'), 5))
