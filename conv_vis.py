'''Visualization of the filters of VGG16, via gradient ascent in input space.

This script can run on CPU in a few minutes (with the TensorFlow backend).

Results example: http://i.imgur.com/4nj4KjN.jpg
'''
from __future__ import print_function
from scipy.misc import imsave
import numpy as np
import time
from keras.applications import vgg16
from keras import backend as K
from functions import your_loss
from keras.models import load_model

K.set_learning_phase(0)
print(K.learning_phase())


# dimensions of the generated pictures for each filter.
img_width = 712
img_height = 712

# the name of the layer we want to visualize
# (see model definition at keras/applications/vgg16.py)


model = load_model('models/segnet_98.h5', custom_objects={'your_loss': your_loss})

# util function to convert a tensor into a valid image


def deprocess_image(y):
    # normalize tensor: center on 0., ensure std is 0.1
    x = y.copy().reshape(712, 712)
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    return x

# build the VGG16 network with ImageNet weights
#model = vgg16.VGG16(weights='imagenet', include_top=False)
#print('Model loaded.')

#model.summary()

# this is the placeholder for the input images
input_img = model.input
print(input_img.get_shape())

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

layers = ['convolution2d_1', 'convolution2d_2', 'convolution2d_3', 'convolution2d_4', 'convolution2d_5', 'convolution2d_6', 'convolution2d_7', 'convolution2d_8']
layers.sort(reverse=True)


for layer in layers:
    kept_filters = []
    filter_len = layer_dict[layer].nb_filter

    for filter_index in range(filter_len):
        # we only scan through the first 200 filters,
        # but there are actually 512 of them
        print('Layer: %s, filter: %d/%d' % (layer, filter_index, filter_len))
        start_time = time.time()

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer_output = layer_dict[layer].output
        loss = K.mean(layer_output[:, filter_index, :, :])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 10.

        # we start from a gray image with some random noise
        #if K.image_dim_ordering() == 'th':
        input_img_data = np.random.random((1, 1, img_width, img_height))
        #else:
        #    input_img_data = np.random.random((1, img_width, img_height, 3))
        #input_img_data = (input_img_data - 0.5) * 20 + 128

        # we run gradient ascent for 20 steps
        for i in range(10):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            print('Current loss value:', loss_value)

        # decode the resulting input image
        if loss_value > 0:
            img = deprocess_image(input_img_data)
            kept_filters.append((img, loss_value))
        end_time = time.time()
        print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

    # we will stich the best 64 filters on a 8 x 8 grid.

    n = int(len(kept_filters)**0.5)

    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top 64 filters.
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    # build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            index = i * n + j
            print(index)
            img, loss = kept_filters[index]
            stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                             (img_height + margin) * j: (img_height + margin) * j + img_height] = img

    # save the result to disk
    imsave('plots/filters/%s.png' % (layer), stitched_filters)

