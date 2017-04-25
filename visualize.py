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

def deprocess_img(y, width, height):
    # normalize tensor: center on 0., ensure std is 0.1
    x = y.copy().reshape(height, width)
    x -= x.mean()
    x /= (x.std() + 1e-5)

    #x *= 0.5

    # clip to [0, 1]
    #x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    #x *= 255
    #if K.image_dim_ordering() == 'th':
    #    x = x.transpose((1, 2, 0))
    #x = np.clip(x, 0, 255).astype('uint8')
    return x


model = load_model('models/12480.h5')
model.summary()
K.set_learning_phase(0)
img_width=56
img_height=56

layer_dict = dict([(layer.name, layer) for layer in model.layers])

print('Layers are:', layer_dict.keys())

input_img = model.layers[0].input
filter_index = 0

layer_output = layer_dict['conv2d_3'].output

# Visualize all filters
for idx, filtr in enumerate(xrange(16)):

    #loss = K.mean(layer_output)
    loss = K.mean(layer_output[:, filter_index, :, :])

    grads = K.gradients(loss, input_img)[0]

    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    iterate = K.function([input_img], [loss, grads])

    #input_img_data = (np.zeros((1, 1, img_width, img_height)))
    input_img_data = np.random.random((1, 1, img_width, img_height))
    #plt.imshow(input_img_data.reshape(img_width, img_height), cmap='Greys')
    #plt.show()
    #plt.imshow(deprocess_img(input_img_data))
    #plt.show()

    step = 1

    cache = None
    for i in range(100):
        loss_value, grads_value = iterate([input_img_data])
        #step, cache = rmsprop(grads_value, cache)
        #input_img_data += step
        input_img_data += grads_value * step
        img2 = deprocess_img(input_img_data, img_width, img_height)
        #plt.imshow(deprocess_img(input_img_data))
        #plt.savefig('viz.png')
        #imsave('viz.png', img2)
        print(i, loss_value, grads_value.mean())
        if loss_value>5:
            imsave('filters/%d.png'%idx, img2)
            break

    #plt.imshow(deprocess_img(input_img_data))
    #plt.show()
