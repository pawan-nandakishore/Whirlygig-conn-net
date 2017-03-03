from keras.models import load_model
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from skimage.color import gray2rgb
from skimage.io import imsave
from datetime import datetime

img = io.imread('images/raw_image_cropped2.png')
gray = io.imread('images/raw_image_cropped2.png', as_grey=True)

model = load_model('models/model.h5')

M, N = gray.shape
print(gray.shape)
c = 21

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

for i in range(c/2, M-c/2):
    for j in range(c/2, N-c/2):
        output = result[count]
        if output>0.5:
            img[i,j] = [255,0,0,255]
            positives += 1

        count += 1

print(count-len(squares2))
plt.imshow(img)
plt.imsave('plots/15_16_batch_aug.png', img)
plt.show()

