from functions import your_loss
from keras.models import load_model
import numpy as np
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.io import imsave

K.set_learning_phase(0)
print(K.learning_phase())

def rmsprop(grads, cache=None, decay_rate=0.95):
    if cache is None:
        cache = np.zeros_like(grads)
    cache = decay_rate * cache + (1 - decay_rate) * grads ** 2
    step = grads / np.sqrt(cache + K.epsilon())

    return step, cache

def deprocess_img(y):
    # normalize tensor: center on 0., ensure std is 0.1
    x = y.copy().reshape(712, 712)
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    #x *= 255
    #if K.image_dim_ordering() == 'th':
    #    x = x.transpose((1, 2, 0))
    #x = np.clip(x, 0, 255).astype('uint8')
    return x


model = load_model('models/segnet_98.h5',custom_objects={'your_loss': your_loss})
model.summary()
K.set_learning_phase(0)
img_width=712
img_height=712

layer_dict = dict([(layer.name, layer) for layer in model.layers])

input_img = model.layers[0].input
filter_index = 1

get_layer_output = K.function([model.layers[0].input, K.learning_phase()], [layer_dict['convolution2d_2'].output])

#layer_output = get_layer_output([input_img, 0])[0]
#import pdb; pdb.set_trace()
layer_output = layer_dict['convolution2d_8'].output

loss = K.mean(layer_output[:, filter_index, :, :])

grads = K.gradients(loss, input_img)[0]

grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

iterate = K.function([input_img], [loss, grads])

input_img_data = (np.zeros((1, 1, img_width, img_height)))
#input_img_data = np.random.random((1, 1, img_width, img_height))
#plt.imshow(input_img_data.reshape(img_width, img_height), cmap='Greys')
#plt.show()
plt.imshow(deprocess_img(input_img_data))
plt.show()

#step = 0.1

cache = None
for i in range(500):
    loss_value, grads_value = iterate([input_img_data])
    step, cache = rmsprop(grads_value, cache)
    input_img_data += step
    #input_img_data += grads_value * step
    img2 = deprocess_img(input_img_data)
    plt.imshow(deprocess_img(input_img_data))
    plt.savefig('viz.png')
    #imsave('viz.png', img2)
    print(i, loss_value, grads_value.mean())

plt.imshow(deprocess_img(input_img_data))
plt.show()
