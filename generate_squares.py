#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:50:35 2017

@author: pawan
"""

import numpy as np 
import matplotlib.pyplot as plt 
import random 
from PIL import Image, ImageDraw
import matplotlib.patches as patches


plt.close()
def get_square(side,edge_pts ):
    """ generates a square of a given side and bottom left points xo,yo
    
    Args:
        side : float side length of a square
        edge_pts : float   bottom left x and y values 
               
    Returns:
        x: float  array shape(4,1) x values of verticies of a square  
        y: float  array shape(4,1) y values of verticies of a square
        
    """
  
    xo = edge_pts[0]
    yo = edge_pts[1]

    x = []
    y =[]
    x3 = xo+side 
    y3 = yo 
    
    x1 = xo
    y1 = yo+side 
    
    x2 = xo+side
    y2 = yo+side

    x = [ xo, x1, x2, x3]
    y = [ yo, y1, y2, y3]
    
    x = np.array(x)
    y = np.array(y)
    
    return(x,y)
        




def create_tuple_list(x_list, y_list ) : 
    """ make a list of tuples from the x and y positions of the vertices of a square 
    
    Args:
        x_list : float list of x coordinates for a square
        y_list : float   list of y coordinates for a square
               
    Returns:
        tuple_list: float  list of tuples where each tuple is an x and y coordinate
                            exampele [(xo,yo), (x1,y1), (x2,y2)]
    """
    
    xylist =np.array([x_list,y_list])
    xylist =tuple(xylist)
    
    tuple_list = []
    for i in range(0,x_list.shape[0]):
        tuple_list.append( tuple((x_list[i],y_list[i] ))) 
        
    
    return(tuple_list)




def generate_mask(canvas_size, polygon, show_mask = True  ): 
    
    """ given a canvas_size convert a polygon to a mask 
    
    Args:
        
        canvas_size : int  shape array [x,y]  contains size of the background 
        polgon      : ints shape list of tuplese  coordinates of a polygon  
        
    Returns:
        
        mask        :  int array  returns a mask on the image of the size [canvas_size, canvas_size]
                                  the size of the mask is determined by the points in polygon 
        
    """
    
    
    img = Image.new('L', (canvas_size, canvas_size), 0)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    mask = np.array(img)
    if(show_mask==True): 
        plt.imshow(mask)
        plt.axis([0, canvas_size, 0, canvas_size])
        plt.show()
    
    return( mask )



def convert_mask_to_rgb(mask, rgb_val): 
    
    """ convert a mask to an rgb image 
    
    Args:
        
        mask        :  int array  binary mask
        rgb_val     :  int array  rgb values, select color for each mask on the basis of these values 
        
    Returns:
        
        img2        :  int array  return an rgb image 
        
    """
    
    
    
    
    rgb_image =np.zeros([mask.shape[0],small_mask.shape[1] ,3], 'uint8')
 
    for i in range(0,3): 
        mask_x, mask_y =np.where(mask==1) 
        rgb_image[mask_x, mask_y, i ] = rgb_val[i]
    
    img2 = Image.fromarray(rgb_image)
    img2 = np.array(img2)
     
    return(img2)
    

def generate_large_mask(mask, mask2): 
    
    """  create an overlap of mask2 over mask 
    
    Args:
        
        mask        :  int array shape [:,:,3] rgb image    
        mask2       :  int array shape [:,:,3] rgb image
        
    Returns:
        
        mask        :  int array shape [:,:,3] rgb image    
        
    """
    
    
    
    
    large_x, large_y = np.where(mask2==1)
    for i in range(0,3): 
        mask[large_x, large_y, i] =rgb_val2[i]
    return (mask)        





def get_tuples_list(s_square_size,l_square_size, s_square_start, l_square_start  ): 
   
    """ convert polygon coordinates into a list of tuples 
    
    Args:
        
        mask        :  int array shape [:,:,3] rgb image    
        mask2       :  int array shape [:,:,3] rgb image
        
    Returns:
        
        mask        :  int array shape [:,:,3] rgb image    
        
    """
   
    #generate the x y positions for both the small square and the large square 
    small_square_x,small_square_y =  get_square(s_square_size, s_square_start)
    large_square_x,large_square_y =  get_square(l_square_size, l_square_start)
    
    
    # make a list of tuples that will be fed into the generate mask function in order to 
    s_square_list = create_tuple_list(small_square_x,small_square_y)
    l_square_list = create_tuple_list(large_square_x,large_square_y)
    
    
    
    return(s_square_list, l_square_list)


def plot_bounding_box(rgb_mask,start_pt,side,start_pt2,side2 ):
    
    
    """plot bounding boxes on masks for the final image
    
    Args:
        
        rgb_mask        :  int array shape [:,:,3] rgb image    
        start_pt        :  int array shape [1,2] starting point for the first bounding box  
        side            :  int scalar            side length of the first bounding box 
        start_pt2       :  int array shape [1,2] starting point for the second bounding box
        side2           :  int scalar            side length of the first bounding box 
        
        
    Returns:
        
        None        :  plot the image rgb_mask and plot two bounding boxes, one for each square   
        
    """

    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    
    ax2.imshow(rgb_mask )
    ax2.add_patch(
        patches.Rectangle(
            (start_pt[0], start_pt[1]),
            side,
            side,
            fill=False,
            edgecolor="y" , # remove background, 
            linewidth = 3
        )
    )
        
    ax2.imshow(rgb_mask )
    ax2.add_patch(
        patches.Rectangle(
            (start_pt2[0], start_pt2[1]),
            side2,
            side2,
            fill=False,
            edgecolor="y" , # remove background, 
            linewidth = 3
        )
    )
        
    plt.axis([0,300,300,0])
    plt.show()
    fig2.savefig('final_labeled_image.png')    



def generate_combined_mask(mask, mask2): 
    
    mask[np.where(mask2==1)] =1
    
   
    plt.imsave('combined_mask.png', mask)
        
    return mask 

###############################################################################
# Base values
small_square_start= [ 100, 100]
large_square_start = [ 120, 120 ]
small_square_size= 50
large_square_size= 120  



# get a list of points for each square, big and small 
small_square_list, large_square_list = get_tuples_list(small_square_size,large_square_size, small_square_start, large_square_start  )



canvas_size = 300

# generate small and large mask 
# both these masks are on a canvas which is of size [canvas_size, canvas_size]
small_mask  = generate_mask(canvas_size, small_square_list, show_mask = False  )
large_mask  = generate_mask(canvas_size, large_square_list, show_mask = False  )

# get a combined mask of two squares 
total_mask = generate_combined_mask(small_mask,large_mask)


rgb_val1= [50,100,250]
rgb_val2= [225,25,15]


# generate small and large mask 
small_color_mask = convert_mask_to_rgb(small_mask, rgb_val1)
large_color_mask = convert_mask_to_rgb(large_mask, rgb_val2)



# get the final labeled image 
final_labeled_image = generate_large_mask(small_color_mask, large_mask)

# get bounding boxes
bbox_small_mask_x, bbox_small_mask_y  =get_square(small_square_size, small_square_start)
bbox_large_mask_x,bbox_large_mask_y   =get_square(large_square_size, large_square_start)


#plot bounding boxes 
plot_bounding_box(final_labeled_image,small_square_start,small_square_size, large_square_start, large_square_size)




