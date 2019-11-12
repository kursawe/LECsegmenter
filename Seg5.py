import os
import numpy as np
import cv2 as cv
from skimage.segmentation import random_walker
from skimage.morphology import skeletonize
from skimage.morphology import watershed as ski_watershed
from skimage import data, util, filters, color
from random import randrange
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema



def removesmallelements(img,minsize):
   #imagem = cv.bitwise_not(img)
    img=img.astype(np.uint8)
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >=minsize:
            img2[output == i + 1] = 255
    return(img2)

def nothing(x):
    pass

     
def watershed(img):

    edges= np.pad(np.ones((510,510)), pad_width=1, mode='constant', constant_values=0)    
    img=img*edges
    img=img.astype(np.uint8)
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(cv.bitwise_not(img),\
    connectivity=4)
    bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    b1,g1,r1 = cv.split(bgr)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    for i in range(0, nb_components):     
        if sizes[i] <=10000 : #MAX CELL SIZE
            b1[output == i + 1]=colors[i][0]
            g1[output == i + 1]=colors[i][1]
            r1[output == i + 1]=colors[i][2]
    image=cv.merge((b1,g1,r1))
    image=cv.erode(image,None) #Optional, to remove membranes (only visual)   
    return image

def simple_watershed(img):

    edges= np.pad(np.ones((510,510)), pad_width=1, mode='constant', constant_values=0)    
    img=img*edges
    img=img.astype(np.uint8)
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(cv.bitwise_not(img),\
    connectivity=4)
    return (output,centroids)

def get_seeds(img):
    global mask
    global cellsizes  
    global output
    global mask_of_centres
    global nb_components
  #  global list_of_seeds
    global centrepixel
    list_of_seeds=[] 
    edges= np.pad(np.ones((510,510)), pad_width=1, mode='constant', constant_values=0)    
    img=img*edges
    img=img.astype(np.uint8)
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(cv.bitwise_not(img),\
    connectivity=4)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    centrepixel=np.zeros((np.shape(output)))
    for i in range(0, nb_components):       
        if sizes[i] <=10000 : #MAX CELL SIZE
            mask=np.zeros((np.shape(output)))          
            mask[output==i+1]=1
            while np.count_nonzero(cv.erode(mask,None) == 1)>=1:
                mask=cv.erode(mask,None)
            loci=np.where(np.isin(mask,1))
            pos_of_random_pixel=loci[0][0],loci[1][0]
            centrepixel[pos_of_random_pixel]=i
           
            list_of_seeds.append(pos_of_random_pixel)
     
    #return list_of_seeds
    return centrepixel
    

def merges(img1,img2,amount):
   # img2=cv.dilate(img2,None) 
    overlay = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    b,g,r = cv.split(overlay)
    r = cv.add(r,amount, dst = r, mask =img2, dtype = cv.CV_8U)
    merged=cv.merge((b,g,r),overlay)
    
    return merged
def GUI(event,x,y,flags,param):
    global Skeletonized_Image
    global drawing, X,Y ,been
    global saved_list, dim
    
    Skeletonized_Image=saved_list[len(saved_list)-1].copy()
   # print(Skeletonized_Image)
    if drawing==True:
        if mode==True:
            if event == cv.EVENT_LBUTTONDOWN:
               debug = cv.line(Skeletonized_Image,(x,y),(X,Y),(255),1).copy()
               saved_list.append(debug)
               drawing=False
               been=False
               
              
               
    if been ==True:
        if drawing==False:
            if mode==True:
                if event == cv.EVENT_LBUTTONDOWN:
                    drawing=True
                    X,Y=x,y
                
    if drawing==False:
        been=True
        
                           
    if mode==False:
        if event== cv.EVENT_LBUTTONDOWN:
            debug=Skeletonized_Image.copy()
            for i in range(-dim,dim):
                for j in range(-dim,dim):
                    
                    debug[y+i,x+j]=0
            saved_list.append(debug)


outcome=[]


colors=[]
for i in range(10000):
    colors.append((randrange(255),(randrange(255)),(randrange(255))))
 
    
    
    
cv.destroyAllWindows() #To reinitialize previous windows (if applicable)




print("Reading files...")

basepath ="/Users/thibautgold/Documents/STACKS"
photos=[]
for entry in os.listdir(basepath): #Read all photos
    if os.path.isfile(os.path.join(basepath, entry)):
        photos.append(entry)
photos.sort()

list_of_means=[]

#Only select the brightest stacks
for tiff_index in range(len(photos)): 
    if photos[tiff_index]!='.DS_Store':
        tiff_photo=cv.imread(basepath+"/"+photos[tiff_index])
        list_of_means.append(np.mean(tiff_photo))
        
array_of_means=np.array(list_of_means)   

local_maxima=argrelextrema(array_of_means, np.greater)[0]
local_minima=argrelextrema(array_of_means, np.less)[0]

false_maximas=[]

local_maxima_list=[]
for maxima in local_maxima:
    local_maxima_list.append(maxima)
    

for minima in local_minima:
    for maxima in local_maxima:
        if minima==maxima+1:
            false_maximas.append(maxima)
            
for false in false_maximas:
    local_maxima_list.remove(false)

false_maximas=[]
for maxima_index in local_maxima_list:
    if list_of_means[maxima_index]<=np.mean(list_of_means):
        false_maximas.append(maxima_index)
          
for false in false_maximas:
    local_maxima_list.remove(false)
        
print("Done")




tiff_images=[]
for Image in local_maxima_list:
    tiff_images.append((cv.imread(basepath+"/"+photos[Image],cv.IMREAD_GRAYSCALE),(cv.imread(basepath+"/"+photos[Image+1],cv.IMREAD_GRAYSCALE))))
    
    
  
cv.namedWindow('Image')
cv.createTrackbar('A','Image',2,15,nothing)   

iter_photo=0
for Image in tiff_images:
    iter_photo+=1
    print ('Image:',iter_photo,"out of:",len(tiff_images))
    
    img=2*(Image[0]+Image[1])
    BLURED= cv.GaussianBlur(img,(5,5),0)
    GAUSSTHRESH=cv.adaptiveThreshold(BLURED,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,0)  
    rem=removesmallelements(GAUSSTHRESH,1000)
    img3 = cv.GaussianBlur(rem,(5,5),0)
    ret,img4 = cv.threshold(img3,118,255,cv.THRESH_BINARY)
    rem=removesmallelements(img4,1000)
    img5 = cv.GaussianBlur(rem,(5,5),0)
    ret,img6 = cv.threshold(img5,53,255,cv.THRESH_BINARY)
    Skeletonized_Image = (skeletonize(img6//255) * 255).astype(np.uint8)
    
    Image_str='Image'+str(iter_photo)
    cv.namedWindow(Image_str)
    cv.createTrackbar('A',Image_str,1,1,nothing)
    cv.setMouseCallback(Image_str,GUI)
    
    mode = True  
    drawing=False 
    been=False
    dim=1 
    saved_list=[]
    saved_list.append(Skeletonized_Image)
    
    while(1):
    
        k = cv.waitKey(1) & 0xFF
        a = cv.getTrackbarPos('A',Image_str)            
        if k == ord('m'):
            mode = not mode
        if k == ord('a'):
            break   
        if k == ord('b'):
            dim+=1 
        if k == ord('s'):
            dim-=1 
        if k== ord('z'):
           if len(saved_list)>1:         
               del saved_list[-1]

        cv.imshow(Image_str,merges(img//2,saved_list[len(saved_list)-1],a*255))

    cv.destroyAllWindows()
    cv.namedWindow('watershed')
    
    watershed_Image=watershed(saved_list[len(saved_list)-1])#.copy()
    outcome.append((saved_list[len(saved_list)-1],watershed_Image))
    
    while(1):
        k = cv.waitKey(1) & 0xFF
        if k == ord('a'):
            break 
        cv.imshow('watershed',watershed_Image)
        
    cv.destroyAllWindows()
    


cv.destroyAllWindows()
print("Processing...")
centroid_array=np.zeros((512,512))
for membrane_skeletons_index in range(len(outcome)):
    simple_watershed_array=simple_watershed(outcome[membrane_skeletons_index][0])[1]
    
    for centroid in simple_watershed_array:
        centroid_array[(int(centroid[0]),int(centroid[1]))]=255//8*membrane_skeletons_index

dilated_centroids=cv.dilate(centroid_array.astype(np.uint8),None)
#dilated_centroids=cv.dilate(centroid_array.astype(np.uint8),None)
#dilated_centroids=centroid_array.astype(np.uint8) 

#clean_dilated_centroids=removesmallelements(dilated_centroids,10)


cv.namedWindow('Image')

while(1):
    k = cv.waitKey(1) & 0xFF
    cv.imshow('Image',merges(img//2,dilated_centroids,255))















