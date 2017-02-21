clear all; 
close all; 


load_image  = imread('whirlygig_boundaries22.png'); 



Boundary_pixels = find((load_image(:,:,2)~=load_image(:,:,3)) );
image2 = load_image; 
sizeval = size(load_image); 
[I,J] = ind2sub(sizeval(1:2),Boundary_pixels) ;
image2 = zeros(sizeval(1:2)); 
image2(Boundary_pixels) = 255;
imagegray = rgb2gray(load_image); 
figure(1)
imshow(imagegray)
labeled1 = bwlabel(255-image2); 
figure(2)
imshow(labeled1) 
hold on 
CC = bwconncomp(image2);
L = labelmatrix(CC);
RGB = label2rgb(L);
% figure, imshow(RGB)
hold off

for label = 1:length(unique(labeled1))-1
    label
    labelled_pixels = find(labeled1==label); 
    grayvals = imagegray(labelled_pixels);
    no_of_zeros = length(find(grayvals<50));
    if (no_of_zeros ==0)
       continue 
    elseif(no_of_zeros> 0) 
        if(double(no_of_zeros)/double(length(grayvals)) > 0.5)
          imagegray(labelled_pixels) =0 ;
          
        end
        
    end
end
imagegray(Boundary_pixels) = 255;
figure(3)
imshow(imagegray) 
background_pixels = find(imagegray==0); 
foreground_pixels = find(imagegray>0); 
inside_pixels = setdiff(foreground_pixels,Boundary_pixels);
image2 = zeros(sizeval(1:2)); 
image2(inside_pixels) =  1; 
figure(3)
imshow(image2)

[Bx,By] = ind2sub(sizeval(1:2),background_pixels) ;
delete_pixels = find(Bx<50); 
Bx(delete_pixels) =[]; 
By(delete_pixels) = []; 
delete_pixels2 = find(By<50); 
Bx(delete_pixels2) =[]; 
By(delete_pixels2) = [];



final_image = zeros(sizeval(1:2)); 

final_image(Boundary_pixels) = 255; 
final_image(inside_pixels) = 127; 
figure
final_image = mat2gray(final_image); 

imshow(final_image)

window_size =28; 
% for i = 1:length(Bx)
%     cutout= imagegray(Bx(i)+window_size:Bx(i)+window_size,By(i)+window_size:By(i)+window_size ); 
%     figure
%     imshow(cutout )
%     filename = ['0' num2str(i)]; 
%     
%     
%     
% end 



% 
% threshval = graythresh(load_image); 
% 
% binary_image = im2bw(load_image, threshval); 
% 
% figure(2)
% imshow(binary_image)
% [B,L] = bwboundaries(binary_image);
% 
% imshow(load_image)
% hold on
% for k = 1:length(B)
%    boundary = B{k};
%    plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 0.2)
% end
% 
% 
