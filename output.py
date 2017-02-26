from keras.models import load_model
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from skimage.color import gray2rgb
from skimage.io import imsave

img = io.imread('raw_image.png', as_grey=True)
img = img[400:1000,400:920]
#rez = np.zeros(img.shape)
colors_heart = gray2rgb(img)

model = load_model('conv.h5')

M, N = img.shape
c = 31

boundaries = []
interiors = []
exteriors = []

#labels = [0, 0.5, 1]
#labelled = {}

#max_out = 0
count = 0
positives = 0

# Run the square over the entire image
for i in range(c/2, M-c/2):
    for j in range(c/2, N-c/2):
        square = img[i-c/2:i+c/2+1, j-c/2:j+c/2+1]
        square = square.reshape(1,1,31,31)
        output = model.predict(square)
       # print(output, max_out)
        #print(output[0].shape)
        #idx = np.argmax(output[0])
        #print (idx)
        #labelled[labels[idx]] = True
        if output>0.5:
            colors_heart[i,j] = [255,0,0]
            positives += 1

        print (positives,i,j)
        #rez[i,j] = labels[idx]
        #print(labelled.keys())
        #print(model.predict(np.array([square])))
            #print(rez)
        count += 1

plt.imshow(colors_heart)
plt.show()

