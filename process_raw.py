import numpy as np
import glob
from skimage.io import imread
from skimage.transform import rotate
import matplotlib.pyplot as plt
from functions import transforms


labels_m = np.load('labeled_images.npy')[:5]

files = glob.glob('raw_images/*')
files.sort()
files = files[0:5]

print(len(files))

labels = np.zeros((labels_m.shape[0], labels_m[0].shape[0], labels_m[0].shape[1], 4))
imgs = [imread(fl, as_grey=True) for fl in files]

plt.imshow(labels_m[0])
plt.show()

labels[labels_m==0]=[0,0,0,1]
labels[labels_m==90]=[0,0,1,0]
labels[labels_m==180]=[0,1,0,0]
labels[labels_m==255]=[1,0,0,0]

xs = []
ys = []

for img, label in zip(imgs, labels):
  print(img.shape[0]+1)
  img_pad = np.zeros((img.shape[0] + 1, img.shape[1] + 1))
  img_pad[:-1,:-1]=img
  img_t = transforms(img_pad)
  #img_t = (img_pad)

  label_pad = np.zeros((label.shape[0]+1, label.shape[1]+1, 4))
  label_pad[:-1,:-1]=label
  label_pad[:,-1,:]=[0,0,0,1]
  label_pad[-1,:,:]=[0,0,0,1]
  label_t = transforms(label_pad)
  #label_t = (label_pad)
  xs.extend(img_t)
  ys.extend(label_t)
  #xs.append(img_t)
  #ys.append(label_t)

xs = np.array(xs)
ys = np.array(ys)

xs = xs.reshape(xs.shape[0], 1, xs.shape[1], xs.shape[2])
ys = ys.reshape(ys.shape[0], ys.shape[1]*ys.shape[2], ys.shape[3])

np.save('data/xs.npy', xs)
np.save('data/ys.npy', ys)
