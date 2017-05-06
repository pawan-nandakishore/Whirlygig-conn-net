import matplotlib.pyplot as plt
from skimage.io import imread
from keras import backend as K
import numpy as np
from scipy.misc import imread
from keras.models import Sequential, load_model
from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import random
import glob
import os
from h5py import h5l
from keras.layers import Lambda
from scipy.misc import imread
from skimage.io import imsave
from tensorflow.python.framework import ops
from tqdm import tqdm
from keras.preprocessing import image
import itertools
import operator
from scipy.signal import convolve

def load_image(path, mode='RGB'):
    """Summary line.

    Extended description of function.

    Args:
        path (int): Read image from path
        mode (str): RGB or L, colored or greyscale

    Returns:
        bool: Description of return value

    """
    img = imread(path, mode=mode).astype(float)/255
    return img

def rotate_thrice(square):
    """ Rotate a square thrice
    
    Args:
        square (ndarray): RGB image to be rotated around
    
    Returns:
        Array of 4 rotated squares
        
    """
    return [square, np.rot90(square, 1), np.rot90(square, 2), np.rot90(square, 3)]

def transforms(square):
    """ Symmetry group of square
    
    Args:
        square (ndarray): RGB image to transform
        
    Returns:
        Array of 8 rotated squares (4 normal, 4 flipped left right)
    """
    return rotate_thrice(square) + rotate_thrice(np.fliplr(square))

def your_loss(y_true, y_pred):
	#weights = np.ones(4)
	#weights = np.array([ 1 ,  1,  1,  1])
	weights = np.array([ 4.2 ,  0.82,  1.3,  0.06])
        #weights = np.array([0.99524712791495196, 0.98911715534979427, 0.015705375514403319])
        #weights = np.array([ 0.91640706, 0.5022308, 0.1])
	#weights = np.array([ 0.05 ,  1.3,  0.55,  4.2])
	#weights = np.array([0.00713773, 0.20517703, 0.15813273, 0.62955252])
	#weights = np.array([1,,0.1,0.001])
	# scale preds so that the class probas of each sample sum to 1
	y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
	# clip
	y_pred = K.clip(y_pred, K.epsilon(), 1)
	# calc
	loss = y_true*K.log(y_pred)*weights
	loss =-K.sum(loss,-1)
	return loss

def raw_to_labels(image):
    """ Convert image to labels
    
    Args:
        image(ndarray): Input rgb 3 channel labeled image
    
    Returns:
        softmax_labeled_image(ndarray): Labeled one hot image with 4 channels
    """
        
    #assert(image.max()==255)
    #if count <= 5:
    junctions_bool = (image[:,:,0]>=150) & ( image[:,:,1] <= 120) & (image[:,:,2] <= 120 )
    inside_bool = (image[:,:,0] <= 120 ) & ( image[:,:,1] <= 120) & (image[:,:,2] >= 130 )
    #else:
    #    inside_bool = (image[:,:,0]>=150) & ( image[:,:,1] <= 120) & (image[:,:,2] <= 120 )
    #    junctions_bool = (image[:,:,0] <= 120 ) & ( image[:,:,1] <= 120) & (image[:,:,2] >= 130 )
    boundary_bool = (image[:,:,0] <=120  ) & ( image[:,:,1] >= 150) & (image[:,:,2] <= 120 )
    exterior_bool = ~inside_bool & ~boundary_bool & ~junctions_bool
    softmax_labeled_image = np.zeros((image.shape[0], image.shape[1], 4))
    softmax_labeled_image[junctions_bool] = [1,0,0,0]
    softmax_labeled_image[boundary_bool] = [0,1,0,0]
    softmax_labeled_image[inside_bool] = [0,0,1,0]
    softmax_labeled_image[exterior_bool] = [0,0,0,1]
    return softmax_labeled_image

def sort_by_number(files):
    """ Sort by the digit of the filename
    
    Args:
        files (list(str)): List of filename to be read in ascending order/
    
    Returns:
        list(str): Sorted filename
    """
    return sorted(files, key=lambda x: int(filter(str.isdigit, x)))

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def register_gradient():
    print(tf)
    """ Register a custom gradient type
    
    Return:
        tensorflow function to set all inhibitor connections to zero and non involved inputs to zero.
    
    """
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)
        
def plot_row(fl, imgs, title='', write=False):
    """ Plots multiple images in the same row and saves them to cmaps/junctions folder
    
    Args:
        imgs (list(ndarray)): List of images
        fl (str); File path to save to
        title (str): Title of plot
        write (bool): Whether to write the plot to a file
        
    Return:
        functional tensorflow expression which sets all other activations to zero.
    
    """
    f, axarr = plt.subplots(1,len(imgs), sharex=True)
    #fig = plt.figure()
    axarr[0].set_title(title)
    
    for i,arr in enumerate(axarr):
        arr.imshow(imgs[i])
    
    if write:
        f.savefig('../plots/cmaps/junctions/%s'%(os.path.basename(fl)))
    
    plt.close(f)
    
def target_category_loss(x, category_index, nb_classes):
    """ Suppress activation of all neurons except category_index
    
    Args:
        x (layer): Keras output layer's activation
        category_index (int): Index of category to keep
        nb_classes (int); Total no of classes
        
    Return:
        functional tensorflow expression which sets all other activations to zero.
    
    """
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    """ Normalizes a tensor """
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def modify_backprop(model, name, modelInitFunc):
    """ Modify default model graph with guided backprop
    
    Args:
        model (keras.models.Model): Input keras model
        name (str): Which gradient type to override with
        modelInitFunc (func): Function which will be called to initialize a new model here
        
    Return:
        new_model (keras.models.Model): Model with overrided gradient operation
    
    """
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = modelInitFunc()
        #new_model = VGG16(weights='imagenet')
    return new_model
        
def guided_backprop_image(model, x, layer='reshape_2'):
    """ Get guided backpropogated gradient image for layer. Objective is maximum activation of layer volume
    
    Args:
        model (keras.models.Model): Input model
        image (tensor): RGB image tensor to propogate gradients to
        layer (str): Layer whose activation to maximize
        
    Returns:
        backprop_img (ndarray): Backpropogated image gradient(preprocessed)
        grads_val (ndaray): Grad values
    """
    
    layer_dict = dict([(lr.name, lr) for lr in model.layers])
    layer_output = layer_dict[layer].output
    
    objective = K.sum(K.max(layer_output, axis=3))
    grads = K.gradients(objective, model.input)[0]
    
    gradient_function = K.function([model.input], [grads])
    
    grads_val = gradient_function([x])[0]
    grads_img = deprocess_image(grads_val.copy())
    
    return grads_img, grads_val

def grad_cam(model, image, category_index, layer_outer, layer_inner):
    """Takes an input model and returns attention heatmap for a particular class.
    Basically allows you to understand wtf is going on in layer_inner. Where is it looking?

    Args:
        model (keras.model): Keras model object
        image (ndarray): Numpy rgb image
        category_index (int): Category index of the final output layer
        layer_outer (str): The layer whose activation is maximized
        layer_inner (str): Which layer w.rt. gradient is taken

    Returns:
        heatmap (ndarray): Grad image (preprocessed)
        cam (ndarray): Grads

    """
    #model = Sequential()
    #model.add(input_model)
    

    #nb_classes = 4
    #target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    #model.add(Lambda(target_layer, output_shape=target_category_loss_output_shape))
    
    #loss = K.sum(model.layers[-1].output)
    
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    #print(layer_dict)
    print('assasas')
    loss = K.sum(layer_dict[layer_outer].output[...,category_index])
    print('assasas')
    
    conv_output = layer_dict[layer_inner].output
    
    grads = K.gradients(loss, conv_output)[0]
    grads = normalize(grads)
    
    gradient_function = K.function([model.input], [conv_output, grads])
    
    output, grads_val = gradient_function([image])
    #print(grads_val)
    output, grads_val = output[0, :], grads_val[0, :, :, :]
    
    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)
    
    #plt.imshow(cam)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (image.shape[1], image.shape[2]))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)
    
    image = image[0, ...]
    image -= np.min(image)
    image = np.minimum(image, 255)
    
    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    
    return heatmap, np.uint8(cam)
    #return heatmap, grads_val#heatmap
    #Return to BGR [0..255] from the preprocessed image
    #image = image[0, :]
    #image -= np.min(image)
    #image = np.minimum(image, 255)

    #cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    #cam = np.float32(cam) + np.float32(image)
    #cam = 255 * cam / np.max(cam)
    #return np.uint8(cam), grads_val
    
def guided_backprop_cam(model, guided_model, image, category_index, layer_outer, layer_inner):
    """Takes an input model and returns the backprop image multiplied by the activation. Tells you where the inner layer volume is looking and what features is it picking up. 

    Args:
        model (keras.model): Keras model object
        model (keras.model): Keras model with backprop activated
        image (tensor): Numpy rgb image tensor
        category_index (int): Category index of the final output layer
        layer_outer (str): The layer whose activation is maximized
        layer_inner (str): Which layer w.rt. gradient is taken

    Returns:
        backprop_cam (ndarray): rgb backprop image
        
    """
    print(layer_inner)
    _, grads_val = guided_backprop_image(guided_model, image, layer_inner)
    
    heatmap, _ = grad_cam(model, image, category_index, layer_outer, layer_inner)
    
    backprop_cam = deprocess_image(grads_val * heatmap[..., np.newaxis])
    return backprop_cam

def guided_cam_evolution(model, x, category_index, lambdaFunc):
    """ Takes a model and shows the layerwise evolution of guided backprop representation
    
    Args:
        model (keras.model): Keras model object
        lamdaFunc (func): Function to instantiate a new model
        
    Returns:
        grad_cams: List of grad cam images for all convolutional layers
        cams: List of all activation maps for each layer
    
    """
    
    register_gradient()
    guided_model = modify_backprop(model, 'GuidedBackProp', lambdaFunc)
        
    def check_layer_2d(layer):
        """ Checks if a layer is 2d shaped """
        name = layer.__class__.__name__
        return name == 'Conv2D' or name == 'MaxPooling2D'
        
    layers = filter(check_layer_2d, model.layers[:-1])
    layer_names = [l.name for l in layers]
        
    cams = [grad_cam(model, x, category_index, model.layers[-1].name, l_name) for l_name in tqdm(layer_names)]
    grad_cams = [guided_backprop_cam(model, guided_model, x, category_index, model.layers[-1].name, l_name) for l_name in tqdm(layer_names)]

    return grad_cams, cams, layer_names

def img_to_tensor(img_path, target_size, preprocess=False):
    """ Loads a tensor from img_path, optionally preprocesses 
    
    Args:
        img_path (str): Image string to load image from
        target_size (tuple): Size incase image can be larger
        
    Returns:
        
    """
    img = image.load_img(img_path, target_size)
    #label = imread(label_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    if preprocess:
        x = preprocess_input(x)
        
    return x
    

def get_data_tensor():
    """ Gets batch of images and labels and returns a tensor by shuffling them
    
    Args:
        imgs(list(ndarray)): List of 3 channel rgb images
        labels(list(ndarray)): List of 4 channel rgb images
        
    Returns:
        xs(tensor)
    
    """
    
def onehot_to_label(onehot):
    """ Converts onehot encoding back to 0,1,2 labels"""
    print("lol not implemented")
    
def tuple_add(a,b):
    """ Adds two tuples """
    return tuple(map(operator.add, a, b))
    
def stride_indices(arr_shape, sq_size, stride):
    """ Return square indices with stride from larger parent square.
    
    Args:
        arr_shape(tuple): Shape of n dimensional image
        sq_size(tuple): What size are the (hyper) squares?
        stride(tuple): What size is each square shifted by?
        
    Returns:
        square_indices(list(list(slices))): List of list of slices/dimension/square 
        
    """
    get_tile_indices = lambda i: xrange(0, arr_shape[i]-sq_size[i]+1, stride[i])
    prods = itertools.product(*map(get_tile_indices, xrange(len(sq_size))))
    
    slicer = lambda x: slice(*x)
    list_slices = lambda prod: map(slicer, zip(prod, tuple_add(prod,sq_size)))
    square_indices = itertools.imap(list_slices, prods)
     
    return square_indices

def squares_to_tiles(arr, sq_size, stride):
    """ Return squares with stride from parent square.
    
    Args:
        arr(ndarray): N dimensional image
        sq_size(ndarray): What size are the (hyper) squares?
        stride(ndarray): WHat size is each square shifted by?
        
    Returns:
        squares(list(ndaray))
        
    """
    indices = stride_indices(arr.shape, sq_size, stride)
    
    squares = [arr[inds] for inds in indices]
    return squares

def tiles_to_square(squares, arr_shape, sq_size, stride):
    """ Converts strided tiles back to an image.
    
    Args:
        squares(list(ndarray)): List of squares to stitch back
        arr_shape(tuple): Shape pf large image to be converted to
        sq_size(ndarray): Size of each square
        stride(ndaray): Stride to shift each square by
        
    Returns:
        arr(ndarray): Large image tiled from all the strided squares
        
    """
    indices = stride_indices(arr_shape, sq_size, stride)
    
    arr = np.zeros(arr_shape)
    weight_square = np.ones(tuple(sq_size))
    
    weights = np.zeros(arr_shape)
    
    for i, inds in enumerate(indices):
        arr[inds] += squares[i]
        weights[inds] += weight_square
        
    return arr/weights

def sample_squares(img, num, kernel, balance=False):
    """ Samples an array of squares from a zstack of images
    
    Args:
        img(ndarray): Image to sample squares from
        num(int): No of images to sample
        kernel(ndarray): Kernel to convolve the image and calculate weights with. Same dimension as img.
        avg_balance(bool): If balancing is needed
        
    Returns:
        squares(ndarray): Stack of rgb images
    
    """
    weighted_kernel = kernel
    
    def balance_kernel(k):
        kernel = k#.copy()
        kernel2 = np.ones((1,kernel.shape[1],kernel.shape[2],1))
        freq = convolve(img, kernel2, mode='valid')
        
        channel_weights = [np.sum(freq[...,c]) for c in xrange(img.shape[-1])]
        #channel_weights = [1,1,1,0.01]
        #print(kernel2.shape)
        
        for c in xrange(img.shape[-1]):
            kernel[...,c] /= channel_weights[c]
        
        #kerne = [np.median(freq[...,c]) ]
        #print(freq.shape, channel_weights, freq[...,0].shape)
        return kernel2
    
    #balanced_kernel = balance_kernel(img)
    
    img_weighted = convolve(img, weighted_kernel, mode='valid')
    img_weighted = img_weighted.reshape(img_weighted.shape[:-1])
    
    img_flat = img_weighted.flatten()
    img_flat /= img_flat.sum()
    img_flat_indices = np.arange(img_flat.shape[0])
    
    choices = np.random.choice(img_flat_indices, num, p=img_flat)
    #import pdb; pdb.set_trace()
    
    def getIndex(choice):
        """ Bring back flattened index to ndarray """
        return np.unravel_index(choice, img_weighted.shape)
    
    def extractSquare(index):
        """ Extract square patch """
        return img[index[-3],index[-2]:index[-2]+kernel.shape[-3], index[-1]:index[-1]+kernel.shape[-2],:]
    
    squares = [extractSquare(getIndex(choice)) for choice in choices]
    return np.array(squares)
    
    