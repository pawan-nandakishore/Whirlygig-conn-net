from keras.models import Sequential, load_model
from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.preprocessing import image
from tensorflow.python.framework import ops
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
#import h5py
K.set_learning_phase(1)
#K._LEARNING_PHASE = tf.constant(0)

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    img_path = path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer='reshape_2'):
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer_output = layer_dict[activation_layer].output
    #max_output = K.max(layer_output, axis=3)
    
    #objective = K.mean(layer_output[:, :, :, channel])
    objective = K.sum(K.max(layer_output, axis=3))
    #max_output = K.max()
    #objective = K.max(layer_dict[activation_layer].output)
    #print(objective.shape)
    grads = K.gradients(objective, model.input)[0]
    #grads = normalize(grads)
    #saliency = K.gradients(max_output, input_img)[0]
    #saliency = K.gradients(K.sum(max_output), input_img)[0]
    #return K.function([input_img], [saliency])

    gradient_function = K.function([model.input], [objective, grads])
    return gradient_function
    #output, grads_val = gradient_function([ima])
    #print(grads_val)
    #output, grads_val = output[0, :], grads_val[0, :, :, :]
    #return grads_val

def modify_backprop(model, name):
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
        new_model = load_model('models/12480.h5')
        #new_model = VGG16(weights='imagenet')
    return new_model

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

def poko():
    a=1
    print('hi')
    print('o')
    
def plot_row(imgs):
    f, axarr = plt.subplots(1,len(imgs), sharex=True)
    
    for i,arr in enumerate(axarr):
        arr.imshow(imgs[i])

def grad_cam(input_model, image, category_index, layer_name):
    #model = Sequential()
    #model.add(input_model)
    

    #nb_classes = 4
    #target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    #model.add(Lambda(target_layer, output_shape=target_category_loss_output_shape))
    
    #loss = K.sum(model.layers[-1].output)
    
    layer_dict = dict([(layer.name, layer) for layer in input_model.layers[1:]])
    #print(layer_dict)
    print('assasas')
    loss = K.sum(layer_dict[layer_name].output[:,:,:,category_index])
    print('assasas')
    
    conv_output = layer_dict['conv2d_8'].output
    
    grads = K.gradients(loss, conv_output)[0]
    grads = normalize(grads)
    
    gradient_function = K.function([model.input], [conv_output, grads])
    
    output, grads_val = gradient_function([image])
    #print(grads_val)
    output, grads_val = output[0, :], grads_val[0, :, :, :]
    
    #plt.imshow(grads_val.sum(axis=2))
    
    #print(output, grads_val)
    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)
    
    #plt.imshow(cam)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (56, 56))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)
    return heatmap, grads_val
    #return heatmap, grads_val#heatmap
    #Return to BGR [0..255] from the preprocessed image
    #image = image[0, :]
    #image -= np.min(image)
    #image = np.minimum(image, 255)

    #cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    #cam = np.float32(cam) + np.float32(image)
    #cam = 255 * cam / np.max(cam)
    #return np.uint8(cam), grads_val

#preprocessed_input = load_image(sys.argv[1])
#model = load_model('models/12480.h5')
#register_gradient()
#guided_model = modify_backprop(model, 'GuidedBackProp')
#layer_name = 'reshape_2'
#img_files = random.sample(glob.glob('cleaned/patches/xs/*.png'), 2)

#for img_fl in img_files:
#    for channel in xrange(1):    
#        print('Channel: %d'%channel)
#        imag = imread(img_fl, as_grey=True)
#        ima = imag.reshape(1,56, 56, 1)
#        saliency_fn = compile_saliency_function(guided_model, channel)
#        
#        output, grads_val = saliency_fn([ima])
#        
#        grad_img = grads_val[0,:,:,0]
#        
#        #heatmap = grad_cam(model, imag, channel, layer_name)
#        
#        #gradcam = grad_img * heatmap
#        
#        grads_val = deprocess_image(grads_val[0, :, :, :].reshape(56,56))
#        #other = deprocess_image(imag)
#        
#        #heatmap_c = cv2.addWeighted(other, 0.5, grads_val, 0.5, 0)
#        
#        #imsave('heatma,heatmap_c)
#        #plt.imshow(grads_val, cmap='Greys')
#        #imsave('heatmaps/%d_%s'%(channel, os.path.basename(img_fl)), heatmap_c)
#        plt.figure()
#        plt.imshow(deprocess_image(grad_img))
#        #plt.figure()
        #plt.imshow(gradcam)
    #plt.hist(grads_val)
    
#x = load_image('cat_dog.png')
#img = np.expand_dims(img, axis=0)
img = imread('cleaned/patches/xs/0.png', as_grey=True)

x = img.reshape(1,56,56,1)

#img = x[0,:,:,:]

#plt.imshow(img)
#plt.show()
#print(img)

#x = np.expand_dims(img, axis=0)
    

#model = VGG16(weights='imagenet')
model = load_model('models/12480.h5')
model.summary()


#predictions = model.predict(x)
#top_1 = decode_predictions(predictions)[0][0]
#print('Predicted class:')
#print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))

#predicted_class = np.argmax(predictions)



#register_gradient()
guided_model = modify_backprop(model, 'GuidedBackProp')





#model = VGG16(weights='imagenet')
lays = ['reshape_2']#, 'block5_conv2', 'block5_conv1', 'block4_conv3']

#for lay in lays:
for channel in xrange(4):
        lay = 'reshape_2'
        heatmap,grads = grad_cam(model, x, channel, lay)
        #g_cam,g_grads = grad_cam(guided_model, x, 0, lay)
        
        #plot_row([cam, g_cam])
        #plot_row([grads.sum(axis=2), g_grads.sum(axis=2)])
        
        saliency_fn = compile_saliency_function(guided_model, lay)
        output, grads_val = saliency_fn([x])
        grads_img = deprocess_image(grads_val)
        
        #plot_row([img, grads_img])
        
        
        #grads_img = deprocess_image(grads_val[0,:,:,:] * heatmap[..., np.newaxis])
        plot_row([img, heatmap])
        #plt.figure()
        #plt.imshow(heatmap)
        #plt.imshow(grads_img)
        #plt.imshow(grads_img)

#print('ishmael')


#plt.imshow(cam)
#output, grads_val = gradient_function([ima])

# Try same approach with segnet

#predictions = model.predict(x)
#top_1 = decode_predictions(predictions)[0][0]
#print('Predicted class:')
#print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))

#saliency_fn = compile_saliency_function(guided_model, 0, 'block5_conv1')
#output, grads_val = saliency_fn([x])
#grads_img = deprocess_image(grads_val[0, :, :, :])
#plt.imshow(grads_img)


#predicted_class = 0
#cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, layer_name)
#cv2.imwrite("gradcam.jpg", cam)


#gradcam = saliency[0] * heatmap[..., np.newaxis]
#cv2.imwrite("guided_gradcam.jpg", deprocess_image(gradcam))
