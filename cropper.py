import glob
from skimage.io import imread, imsave
import matplotlib.pyplot as plt

raws = sorted(glob.glob('raw/*'))
labeled = sorted(glob.glob('labeled/*'))

coords = [(750, 520), (780, 520), (760, 500), (780, 450), (750, 500)]

def crop(img, y, x):
  width = 320
  img2 = img.copy()
  return img2[x:x+width,y:y+width]

# Can add reading as grey here
count = 0

for raw, label in zip(raws, labeled):
  raw_img = crop(imread(raw), *coords[count]) 
  label_img = crop(imread(label), *coords[count])

  imsave('small_cropped/'+raw+'.png', raw_img)
  imsave('small_cropped/'+label+'.png', label_img)

  count += 1
