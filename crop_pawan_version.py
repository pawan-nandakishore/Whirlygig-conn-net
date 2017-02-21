import numpy as np
from scipy.io import loadmat
import random
import matplotlib.pyplot as plt
import os
import cv2 


# Also try unit test with 5 n 3 linear convolution
df = loadmat('final_image_with_classes.mat')
img = df['final_image']


raw_image = cv2.imread('raw_image.png')
gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

M, N = img.shape
c = 5
sample_size = 10000

boundaries = []
interiors = []
exteriors = []
boundaries_pixellist = []
interiors_pixellist = []
exteriors_pixellist = []




def generateSamples(data, foldername, num,fileindx,pixellist):
    random_index_list = np.random.random_integers(1,len(pixellist),num) 
    print(random_index_list)

    data2 = np.asanyarray(data)
    samples = data2[random_index_list] 
    pixellist2 = np.asanyarray(pixellist)    
    files = pixellist2[random_index_list]
  
    newpath = r'/media/pawan/0B6F079E0B6F079E/PYTHON_SCRIPTS/Data science challenges/whirlygig/' + foldername 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    for i in range(0,num):
        sample = samples[i] 
        filename = str(fileindx) +'_'+str(i) +'_'+ files[i]
        filename = os.path.join(newpath,filename)
        np.save(filename,sample) 




# Run the square over the entire image

for i in range(c/2, M-c/2):
    for j in range(c/2, N-c/2):
        square = gray_image[i-c/2:i+c/2+1, j-c/2:j+c/2+1]
        if gray_image[i][j] == 0:
            exteriors.append(square)
            pixel_label = str(i)+'_'+str(j)
            exteriors_pixellist.append(pixel_label)
        elif img[i][j] == 1:
            boundaries.append(square)
            pixel_label = str(i)+'_'+str(j)
            boundaries_pixellist.append(pixel_label)

        else:
            interiors.append(square)
            pixel_label = str(i)+'_'+str(j)
            interiors_pixellist.append(pixel_label)
        
        
        
#for i in xrange(c/2, N-c/2)
boundaryindex = 1 
interiorindex= 2 
exteriorindex =0 

foldername1 = 'boundaries_' + str(c) +'x'
foldername2 = 'exteriors_' + str(c) +'x'
foldername3 = 'interiors_' + str(c) +'x'



generateSamples(boundaries, foldername1, sample_size,1,boundaries_pixellist)
generateSamples(interiors, foldername2, sample_size,2,interiors_pixellist)
generateSamples(exteriors, foldername3, sample_size,0,exteriors_pixellist)
