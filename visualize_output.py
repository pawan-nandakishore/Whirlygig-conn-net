from skimage.io import imsave
import numpy as np
import matplotlib.pyplot as plt

size = 432

xs = np.load('xs_e.npy')
ys = np.load('ys_e.npy')

count = 0

for x,y in zip(xs, ys):
    y = y.reshape(size, size, 3)

    x_rgba = np.zeros((size,size,4))
    y_rgba = np.zeros((size,size,4))

    for i in xrange(size):
        for j in xrange(size):
            y_rgba[i,j,np.argmax(y[i,j])] = 1

    x_rgba[:,:,:-1]=x.reshape(size, size, 3)
    x_rgba[:,:,3] = 1
    y_rgba[:,:,3] = 1

    #plt.imshow(x_rgba)
    #plt.show()

    imsave(('plots/outs/%d_i.png')%count, x_rgba)
    imsave(('plots/outs/%d_o.png')%count, y_rgba)

    count += 1
