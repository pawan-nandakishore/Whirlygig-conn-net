
import scipy
from scipy.misc import imread,imsave
from scipy import ndimage
import matplotlib.pyplot as plt 
import numpy as np
import os
import numpy.matlib as npmat 



def get_num_pixels(list_of_labels, labeled):
    list_of_centroids = [] 
    for label in range(1,np.max(list_of_labels)) : 
        label_bool = (labeled==label)
        pixels = np.where(label_bool)
        num_pixels[label] = len(pixels[0])
        
        points = np.where(label_bool==1)
        x = points[0]
        y = points[1]
        centroid = [sum(x) / len(points[0]), sum(y) / len(points[0])]
        centroid = np.round((centroid))
        centroid = tuple(centroid)
        list_of_centroids.append(centroid)
            

    return num_pixels, list_of_centroids    



def calculate_outside_interval(values_num_pixels): 
    std_val = np.std(values_num_pixels)
    confidence = 0.95*std_val 
    confidence_interval = [np.mean(values_num_pixels) - confidence,np.mean(values_num_pixels) + confidence]
    values_num_pixels= np.array(values_num_pixels )
    
    errval = np.where((values_num_pixels   <confidence_interval[0]))
    errval2 = np.where((values_num_pixels   >confidence_interval[1]))
    errval_all = np.hstack((errval,errval2))
    return(errval_all)


def get_cropped_images_list(image, labeled,errval_all, list_of_centroids): 
    for l in range(1,errval_all.shape[1]) : 
        

        label = errval_all[0][l]
        # label_bool = (labeled==label)
        
        #get list of centroids
        centroid = list_of_centroids[label-1]   
        
        # crop image 
        image2 = image
        cropped_image = image2[centroid[0]-image_limit:centroid[0]+image_limit,centroid[1]-image_limit:centroid[1]+image_limit,: ]
        cropped_images.append(cropped_image)
        
        # plt.scatter(centroid[1],centroid[0], color = 'green' )
        # plt.text(centroid[1],centroid[0] ,str(num_pixels[label]), color = 'blue',fontsize=15)
    return(cropped_images )       

def get_image_patch(image,cropped_images ): 

    stride =image.shape[1]/50 
    stride = np.linspace(50, mask.shape[1]-50, 20)
    positions= []
    print(stride)
    stride = np.round(stride,0)

    for i in stride: 
        for j in stride: 
            positions.append(tuple((i,j)))
            # plt.scatter(i,j,color='red')
    
    length_ij = len(positions)

    print(len(cropped_images))
    ij = 0  
    for i in stride: 
        for j in stride: 
            
            single_image = cropped_images[ij] 
            imx_m=np.round((i-single_image.shape[1]/2))
            imy_m =np.round((j-single_image.shape[1]/2))
            imx_p=np.round((i+single_image.shape[1]/2))
            imy_p =np.round((j+single_image.shape[1]/2))
            
            empty_image[imx_m:imx_p,imy_m:imy_p] = single_image

            ij += 1 
            if (ij==len(cropped_images)): 
                ij = 0 

         
        
    return(empty_image,positions)



def open_files(i,file_ind,folderloc,raw_folder,raw_files,labeled_folder,labeled_files):
 # For each image in the folder create mask  
    imageLoc = folderloc + i
    raw_imageLoc = raw_folder+raw_files[file_ind]
    labeled_imageLoc = labeled_folder+labeled_files[file_ind]

    file_ind += 1 
    
    image = imread(imageLoc, mode='RGB')
    raw_image = imread(raw_imageLoc, mode='RGB')
    labeled_image = imread(labeled_imageLoc, mode='RGB')
    
    print(image.shape,raw_image.shape)

    mask = np.zeros(image.shape[0:2])
    
    mask_bool  = (image[:,:,0]!=image[:,:,2])
    mask[mask_bool] = 255
    return(mask,image,raw_image,labeled_image,mask_bool)

######################################################################## 
#########################MAIN FUNCTION ####################################
##################################################################################



raw_folder = 'cleaned/raw/'
labeled_folder = 'cleaned/labeled/'
folderloc = 'plots/results/'
files = os.listdir(folderloc)
raw_files= os.listdir(raw_folder)
labeled_files = os.listdir(labeled_folder)
print(files, raw_files, labeled_files)
file_ind = 0 

print(files)

for i in files:
   
    plt.close('all')

    mask,image,raw_image,labeled_image,mask_bool= open_files(i,file_ind,folderloc,raw_folder,raw_files,labeled_folder,labeled_files)
    image = image-255 
    # label all the individual images and get a list of labels
    # plt.figure()
    # plt.imshow(labeled_image)
    # plt.show()
    
    labeled, nr_objects = ndimage.label(mask)
    list_of_labels = np.unique(labeled) 
    

    num_pixels = dict()
    num_pixels[0] = 0 
    image_limit =  15
    
    #get list of image centroids as a dict, get the values in the dict 
    num_pixels, list_of_centroids = get_num_pixels(list_of_labels, labeled)
    values_num_pixels = num_pixels.values()
      

    errval_all = calculate_outside_interval(values_num_pixels)

    empty_image  = np.ones(image.shape)*255

    cropped_images =[]
    image_type = labeled_image
    
    cropped_images = get_cropped_images_list(image_type, labeled,errval_all, list_of_centroids)
    
    # placing the patches in empty image 
    empty_image,positions  = get_image_patch(image_type,cropped_images )
    
    plt.figure()
    plt.imshow(empty_image/255)
    plt.show()
    plt.figure()
    plt.imshow(labeled_image)
    plt.show()
    
    imsave('plots/empty.png', empty_image)




    
    
    
    break
