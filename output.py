from keras.models import load_model
import matplotlib.pyplot as plt
from skimage import io
import numpy as np

img = io.imread('raw_image.png', as_grey=True)
rez = np.zeros(img.shape)

model = load_model('full.h5')

M, N = img.shape
c = 5

boundaries = []
interiors = []
exteriors = []

labels = [0, 0.5, 1]
labelled = {}

# Run the square over the entire image
for i in range(c/2, M-c/2):
    for j in range(c/2, N-c/2):
        print(i,j)
        square = img[i-c/2:i+c/2+1, j-c/2:j+c/2+1].flatten()*255
        output = model.predict(np.array([square]))
        #print(output[0].shape)
        idx = np.argmax(output[0])
        #print (idx)
        labelled[labels[idx]] = True
        rez[i,j] = labels[idx]
        #print(labelled.keys())
        #print(model.predict(np.array([square])))
        if i%100==0:
            #print(rez)
            #plt.imshow(rez)
            #plt.show()
            io.imsave('fig.tiff', rez)

