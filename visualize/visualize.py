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
from functions import load_image
from keras import backend as K
import random
import glob
from tqdm import tqdm
K.set_learning_phase(0)

def visualize2():
    """ Have to come up with a better name for this.
        Generate attention gifs. Temporal dynamics will be much more powerful
    """
    img_files = glob.glob('../images/patches/xs/*.png')
    img_files = random.sample(img_files, 500)
    
    modelGen = lambda : load_model('../models/12480_rgb.h5')
    model = modelGen()
    
    register_gradient()
    guided_model = modify_backprop(model, 'GuidedBackProp', modelGen)
    
    for img_fl in tqdm(img_files):
        visualize(img_fl, model, guided_model)
        

def visualize(img_fl, model, guided_model):
    """ What? What? WHat?
    """
    
    img = load_image(img_fl, mode='RGB')
    #print(img.shape)
    #plt.imshow(img)
    
    x = np.expand_dims(img, axis=0)
    #print(x.shape)
    #x = preprocess_input(x)
    
    #out = model.predict(x)
    pred_class = 0
    
    final = guided_backprop_cam(model, guided_model, x, pred_class, 'reshape_2', 'conv2d_8')
    #plt.imshow(final)
    
    plot_row(img_fl, [img, final])
    #print(final.mean(), label.mean())

    #np.testing.assert_array_equal(final, label)

if __name__ == "__main__":
    visualize2()
