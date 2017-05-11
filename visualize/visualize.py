# For every rose has its thorns
import sys
sys.path.append('..')

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from h5py import h5l
from functions import guided_backprop_cam, guided_backprop_image, register_gradient, modify_backprop, plot_row, grad_cam
import matplotlib.pyplot as plt
from keras.models import load_model
#from scipy.misc import imread
from functions import load_image, sort_by_number
from keras import backend as K
import random
import glob
from tqdm import tqdm
import re
K.set_learning_phase(0)
from skimage.io import imread, imsave

def visualize2():
    """ Have to come up with a better name for this.
        Generate attention gifs. Temporal dynamics will be much more powerful
    """
    img_files = sort_by_number(glob.glob('../plots/comparisons/errors_i/*.png'))
    img_files = random.sample(img_files, min(64, len(img_files)))
    
    modelGen = lambda : load_model(sorted(glob.glob('../models/*'), key=lambda name: int(re.search(r'\d+', name).group()), reverse=True)[0])
    model = modelGen()
    
    register_gradient()
    guided_model = modify_backprop(model, 'GuidedBackProp', modelGen)
    
    for i,img_fl in tqdm(enumerate(img_files)):
        backprop, heatmap, cam = visualize(img_fl, model, guided_model)
        plt.imsave('../plots/cams/%d.png'%i, cam)
        plt.imsave('../plots/heatmaps/%d.png'%i, heatmap)
        plt.imsave('../plots/backprops/%d.png'%i, backprop)

def visualize(img_fl, model, guided_model):
    """ What? What? WHat?
    """
    
    img = load_image(img_fl, mode='RGB')
    
    x = np.expand_dims(img, axis=0)
    
    #out = model.predict(x)
    pred_class = 0
    
    final = guided_backprop_cam(model, guided_model, x, pred_class, 'reshape_2', 'conv2d_8')
    heatmap, cam = grad_cam(model, x, pred_class, 'reshape_2', 'conv2d_8')
    #plt.imshow(final)
    
    #print(final.mean(), label.mean())
    return final, heatmap, cam

    #np.testing.assert_array_equal(final, label)

if __name__ == "__main__":
    visualize2()
