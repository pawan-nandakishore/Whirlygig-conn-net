
import scipy
from scipy.misc import imread,imsave
from scipy import ndimage
import matplotlib.pyplot as plt 
import numpy as np
import os
import numpy.matlib as npmat 
import random 


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


def get_cropped_images_list(input_image, labeled,errval_all, list_of_centroids): 
    all_new_centroids =  []
    for l in range(1,errval_all.shape[1]) : 
        

        label = errval_all[0][l]
        # label_bool = (labeled==label)
        
        #get list of centroids
        centroid = list_of_centroids[label-1]   
        all_new_centroids.append(centroid)
        # crop image 
        
        cropped_image = input_image[centroid[0]-image_limit:centroid[0]+image_limit,centroid[1]-image_limit:centroid[1]+image_limit,: ]
        cropped_images.append(cropped_image)
        
        # plt.scatter(centroid[1],centroid[0], color = 'green' )
        # plt.text(centroid[1],centroid[0] ,str(num_pixels[label]), color = 'blue',fontsize=15)
    return(cropped_images, all_new_centroids)       

def get_cropped_images_other(input_image,  new_list_of_centroids,list_cropped_images): 
    
    for l in range(1,len(new_list_of_centroids)) : 

        centroid = new_list_of_centroids[l]
        # crop image 
        
        cropped_image_single = input_image[centroid[0]-image_limit:centroid[0]+image_limit,centroid[1]-image_limit:centroid[1]+image_limit,: ]
        list_cropped_images.append(cropped_image_single)
        
        
    return(list_cropped_images)       



def get_image_patch(input_image,list_cropped_images2,input_empty ): 

    stride =input_image.shape[1]/50 
    stride = np.linspace(50, mask.shape[1]-50, 20)
    positions= []
    print(stride)
    stride = np.round(stride,0)

    for i in stride: 
        for j in stride: 
            positions.append(tuple((i,j)))
            # plt.scatter(i,j,color='red')
    
    length_ij = len(positions)

    print(len(list_cropped_images2))
    ij = 0  
    for i in stride: 
        for j in stride: 
            
            single_image = list_cropped_images2[ij] 
            imx_m=np.round((i-single_image.shape[1]/2))
            imy_m =np.round((j-single_image.shape[1]/2))
            imx_p=np.round((i+single_image.shape[1]/2))
            imy_p =np.round((j+single_image.shape[1]/2))
            
            input_empty[imx_m:imx_p,imy_m:imy_p] = single_image

            ij += 1 
            if (ij==len(list_cropped_images2)): 
                ij = 0 

         
        
    return(input_empty,positions)



def open_files(i,input_image):
 # For each image in the folder create mask  
    # imageLoc = folderloc + i
    # raw_imageLoc = raw_folder+raw_files[file_ind]
    # labeled_imageLoc = labeled_folder+labeled_files[file_ind]
    
    
    # image = imread(imageLoc, mode='RGB')
    # raw_image = imread(raw_imageLoc, mode='RGB')
    # labeled_image = imread(labeled_imageLoc, mode='RGB')
    
    # print(image.shape,raw_image.shape)

    mask = np.zeros(image.shape[0:2])
    
    mask_bool  = (image[:,:,0]!=image[:,:,2])
    mask[mask_bool] = 255
     
    
    return(mask,mask_bool)

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
all_cropped_images  = []
raw_cropped_images = []
labeled_cropped_images = []
filecounter = 0 

all_result_images = [ imread(folderloc+jimage, mode='RGB')  for jimage in files]
all_raw_images = [ imread(raw_folder+rimage, mode='RGB')  for rimage in raw_files]
all_labeled_images = [ imread(labeled_folder+limage, mode='RGB')  for limage in labeled_files]
plt.close('all')
   
for single_img in range(0,len(all_result_images)):
    
    labeled_image = all_labeled_images[single_img]
    raw_image = all_raw_images[single_img]
    image = all_result_images[single_img]
   
    
    mask, mask_bool= open_files(single_img, image)
    


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
    empty_image2  = np.ones(image.shape)*255
    empty_image3  = np.ones(image.shape)*255
    cropped_images =[]
    cropped_images2 =[]
    cropped_images3 = []

    
    #original cropped images
    image = image-255 

    cropped_images,all_new_centroids = get_cropped_images_list(image, labeled,errval_all, list_of_centroids)
    all_cropped_images.append(cropped_images) 
    empty_image,positions  = get_image_patch(image,cropped_images,empty_image)
   
    
    cropped_images2 = get_cropped_images_other(raw_image,  all_new_centroids,cropped_images2)
    empty_image2,positions  = get_image_patch(raw_image,cropped_images2,empty_image2)
    raw_cropped_images.append(cropped_images2) 
   


    sample_cropped = random.sample(cropped_images2,10)
    print("all new cent:", len(all_new_centroids))
    if(filecounter==1):
        # f, axarr = plt.subplots(2, sharey = True)
        # axarr[0].imshow(mask)
        # for cent in range(len(all_new_centroids)):
        #     centroid= all_new_centroids[cent] 
        #     axarr[0].scatter(centroid[1],centroid[0],color ='red') 
        
        # axarr[1].imshow(raw_image)

        # for cent in range(len(all_new_centroids)):
        #     centroid= all_new_centroids[cent] 
        #     axarr[1].scatter(centroid[1],centroid[0],color ='green') 
        
        # plt.show() 


        # f, axarr = plt.subplots(3,3)
        # imval  = 0 
        # for k in range(0,3):
        #     for k2 in range(0,3):
        #         axarr[k,k2].imshow(sample_cropped[imval])
        #         imval+=1
        
        plt.figure('raw')
        plt.imshow(empty_image2)
        plt.show()
        f, axarr = plt.subplots(2, sharey = True)
        axarr[0].imshow(empty_image)
        axarr[1].imshow(empty_image2)
        plt.show()
        break
    
    
    #labeled_cropped_images

    
    cropped_images3 = get_cropped_images_other(labeled_image,  all_new_centroids,cropped_images3)
    empty_image3,positions  = get_image_patch(labeled_image,cropped_images3,empty_image3)
    

    

    labeled_cropped_images.append(cropped_images3) 
    
    filecounter += 1 

    # placing the patches in empty image 


# sample_cropped = random.sample(cropped_images,10)

# f, axarr = plt.subplots(3,3)
# imval  = 0 
# for i in range(0,3):
#     for j in range(0,3):
#         axarr[i,j].imshow(sample_cropped[imval])
#         imval+=1

plt.show()

print('len of labeled: ',len(np.array(labeled_cropped_images)))

print('len of raw: ',len(np.array(raw_cropped_images)))

print('len of image: ',len(np.array(all_cropped_images)))



 # empty_image,positions  = get_image_patch(image_type,cropped_images )
    
# plt.figure()
# plt.imshow(empty_image/255)
# plt.show()
# plt.figure()
# plt.imshow(labeled_image)
# plt.show()
    
# imsave('plots/empty.png', empty_image)




    
    
    
    