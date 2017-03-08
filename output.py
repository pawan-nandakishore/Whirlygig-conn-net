from keras.models import load_model
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from skimage.color import gray2rgb
from skimage.io import imsave
from datetime import datetime

img = io.imread('images/fucked.png')
gray = io.imread('images/fucked.png', as_grey=True)

model = load_model('models/model.h5')

M, N = gray.shape
print(gray.shape)
c = 17

positives = 0

squares = []

startTime = datetime.now()

# Run the square over the entire image
for i in range(c/2, M-c/2):
    for j in range(c/2, N-c/2):
        square = gray[i-c/2:i+c/2+1, j-c/2:j+c/2+1]
        squares.append(square.reshape(1,c,c))

squares2 = np.array(squares)
result = model.predict(squares2)
print(datetime.now() - startTime)

count = 0

zeros = np.zeros(img.shape)

for i in range(c/2, M-c/2):
    for j in range(c/2, N-c/2):
        output = result[count]
        zeros[i,j,np.argmax(output)] = 255
        count += 1

zeros[:,:,3]=255

print(count-len(squares2))
plt.imshow(zeros/255)
plt.imsave('plots/new.png', zeros)
plt.show()

