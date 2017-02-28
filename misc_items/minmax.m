% Function  [value eindx] = minmax(pt1,array,indx) where indx =1 applies
% min and indx = 0 applies max to the difference between pt1 and array 


function   [value eindx] = minmax(pt1,array,indx) 
              
          

         diffval=  repmat(pt1,length(array),1)-array; 
         if(indx== 1)
             
           [value eindx] =  min(abs(diffval)); 
         elseif(indx==0)
             [value eindx] =  max(abs(diffval));
             
             
         end