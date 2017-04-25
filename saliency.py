import cv2
import os
import glob
import numpy as np
import random
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from skimage.color import rgb2gray, gray2rgb
from PIL import Image
from keras.layers.convolutional import Convolution2D

from vis.utils.utils import stitch_images
from vis.losses import ActivationMaximization
from vis.optimizer import Optimizer
from vis.utils import utils
from vis.regularizers import TotalVariation, LPNorm
from matplotlib.pylab import cm


from keras.layers.pooling import _Pooling2D

#from vis.visualization import visualize_saliency, visualize_cam

from keras.models import load_model

# Build the VGG16 network with ImageNet weights
#model = VGG16(weights='imagenet', include_top=True)
model = load_model('models/12480.h5')
model.summary()

# Try out saliency example now
img_files = random.sample(glob.glob('cleaned/patches/xs/*.png'), 500)


def visualize_cam(model, layer_idx, filter_indices,
                  seed_img, penultimate_layer_idx=None,
                  text=None, overlay=True):

    # Search for the nearest penultimate `Convolutional` or `Pooling` layer.
    if penultimate_layer_idx is None:
        for idx, layer in utils.reverse_enumerate(model.layers[:layer_idx-1]):
            if isinstance(layer, (Convolution2D, _Pooling2D)):
                penultimate_layer_idx = idx
                break

    if penultimate_layer_idx is None:
        raise ValueError('Unable to determine penultimate `Convolution2D` or `Pooling2D` '
                         'layer for layer_idx: {}'.format(layer_idx))
    assert penultimate_layer_idx < layer_idx

    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), 1)
    ]

    penultimate_output = model.layers[penultimate_layer_idx].output
    opt = Optimizer(model.input, losses, wrt=penultimate_output)
    _, grads, penultimate_output_value = opt.minimize(seed_img, max_iter=1, verbose=False)

    # We are minimizing loss as opposed to maximizing output as with the paper.
    # So, negative gradients here mean that they reduce loss, maximizing class probability.
    print('lol')
    grads *= -1

    # Average pooling across all feature maps.
    # This captures the importance of feature map (channel) idx to the output
    s_idx, c_idx, row_idx, col_idx = utils.get_img_indices()
    weights = np.mean(grads, axis=(s_idx, row_idx, col_idx))

    # Generate heatmap by computing weight * output over feature maps
    s, ch, rows, cols = utils.get_img_shape(penultimate_output)
    heatmap = np.ones(shape=(rows, cols), dtype=np.float32)
    for i, w in enumerate(weights):
        heatmap += w * penultimate_output_value[utils.slicer[0, i, :, :]]

    # The penultimate feature map size is definitely smaller than input image.
    s, ch, rows, cols = utils.get_img_shape(model.input)
    heatmap = cv2.resize(heatmap, (cols, rows), interpolation=cv2.INTER_CUBIC)

    # ReLU thresholding, normalize between (0, 1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Convert to heatmap and zero out low probabilities for a cleaner output.
#    heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
#    #heatmap_colored[np.where(heatmap <= 0.2)] = 0
#
#    if overlay:
#        heatmap_colored = cv2.addWeighted(seed_img, 1, heatmap_colored, 0.5, 0)
#    if text:
#        cv2.putText(heatmap_colored, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
#    return heatmap_colored
    

for img_fl in img_files:

    img = imread(img_fl, as_grey=True)
    img2 = imread(img_fl, as_grey=False)
    img = img.reshape(img.shape[0], img.shape[1], 1)
    layer_name = 'reshape_2'
    layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]
    #layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    filters = [0]

    im2 = gray2rgb(img.reshape(56,56))

    #heatmap = visualize_saliency(model, layer_idx, filters, img, overlay=False)
    heatmap_c = visualize_cam(model, layer_idx, filters, img, overlay=False)
    #print(img2.shape, heatmap.shape)
    #heatmap = cv2.addWeighted(img2, 0.5, heatmap, 1, 0)
    heatmap_c = cv2.addWeighted(img2, 0.5, heatmap_c, 0.5, 0)
    #plt.imshow(heatmap_c)
    #imsave('heatmaps/%s.png'%os.path.basename(img_fl), heatmap)
    imsave('heatmaps/%s_r.png'%os.path.basename(img_fl), heatmap_c)

    #losses = [
    #    (ActivationMaximization(layer_dict[layer_name], output_class), 2),
    #    (LPNorm(model.input), 10),
    #    (TotalVariation(model.input), 10)
    #]
    #opt = Optimizer(model.input, losses)

    #opt.minimize(max_iter=500, verbose=True,
    #             progress_gif_path='opt_progress')

