#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 16:45:33 2019

@author: Thibaut Goldsborough, tg76@st-andrews.ac.uk
"""

import numpy as np
import cv2 as cv
from skimage.morphology import skeletonize
from random import randrange
#from PIL import Image
from matplotlib import pyplot as plt
#%matplotlib auto
colors=[]
for i in range(10000):
    colors.append((randrange(255),(randrange(255)),(randrange(255))))
 
    
cv.destroyAllWindows()


def removesmallelements(img,minsize):
   #imagem = cv.bitwise_not(img)
    img=img.astype(np.uint8)
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= minsize:
            img2[output == i + 1] = 255
    return(img2)

def nothing(x):
    pass

     
def watershed(img):
    global cellsizes  
    cellsizes=[]  
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
            cellsizes.append(np.size(b1[output == i + 1]))
            b1[output == i + 1]=colors[i][0]
            g1[output == i + 1]=colors[i][1]
            r1[output == i + 1]=colors[i][2]
    image=cv.merge((b1,g1,r1))
    image=cv.erode(image,None) #Optional, to remove membranes (only visual)   
    return image

    

def merges(img1,img2,amount):
   # img2=cv.dilate(img2,None) 
    overlay = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    b,g,r = cv.split(overlay)
    r = cv.add(r,amount, dst = r, mask =img2, dtype = cv.CV_8U)
    merged=cv.merge((b,g,r),overlay)
    
    return merged

#img = cv.imread('/Users/thibautgold/Documents/Paphphotos/cellsstack.jpg',cv.IMREAD_GRAYSCALE)
stack1 = cv.imread('/Users/thibautgold/Documents/Paphphotos/img1.jpg',cv.IMREAD_GRAYSCALE)
stack2 =cv.imread('/Users/thibautgold/Documents/Paphphotos/img2.jpg',cv.IMREAD_GRAYSCALE)

img=2*(stack1+stack2)
#image=np.maximum(stack1,stack2)


cv.namedWindow('skeleton')
cv.namedWindow('watershed')


cv.createTrackbar('A','skeleton',118,200,nothing)
cv.createTrackbar('B','skeleton',53,200,nothing)
cv.createTrackbar('C','skeleton',5,20,nothing)
cv.createTrackbar('D','skeleton',1,1,nothing)
cv.createTrackbar('E','skeleton',1,40,nothing)

myfigure=plt.figure()
largewindow=np.zeros((2*img.shape[0],3*img.shape[1]))
image_object=plt.imshow(largewindow)

plt.ion()
while(1):

    k = cv.waitKey(1) & 0xFF



    a = cv.getTrackbarPos('A','skeleton') #118
    b = cv.getTrackbarPos('B','skeleton') #53
    c = cv.getTrackbarPos('C','skeleton')
    d = cv.getTrackbarPos('D','skeleton')
    e = cv.getTrackbarPos('E','skeleton')
    if c<=1:
        c=2
    #clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(50,50))
    #hist=cv.equalizeHist(img)
   # cl1 = clahe.apply(img)
   # plt.figure()
    #img2=cv.medianBlur(img,2*e+1,None)
    #img2	=cv.bilateralFilter(img,e,e,e,e,0)

    img2 = cv.GaussianBlur(img,(3*e,3*e),e)
    GAUSSTHRESH=cv.adaptiveThreshold(img2,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,2*c+1,0)  
   # GAUSSTHRESH=cv.adaptiveThreshold(img2,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,2*c+1,0)
    rem=removesmallelements(GAUSSTHRESH,1000)
    img3 = cv.GaussianBlur(rem,(5,5),0)
    ret,img4 = cv.threshold(img3,a,255,cv.THRESH_BINARY)
    rem=removesmallelements(img4,1000)
    img5 = cv.GaussianBlur(rem,(5,5),0)
    ret,img6 = cv.threshold(img5,b,255,cv.THRESH_BINARY)
   # img7= cv.morphologyEx(img6, cv.MORPH_OPEN, kernel=None)
   # img4 = cv.Laplacian(img3,cv.CV_64F)
    #MEANTHRESH= cv.adaptiveThreshold(img2,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,a)
   # rem=removesmallelements(img7,d)
    Skeletonized_Image = (skeletonize(img6//255) * 255).astype(np.uint8)
    Merge=merges(img//2,Skeletonized_Image,d*255)
    Watershed=watershed(Skeletonized_Image)
   # cv.imshow('skeleton',Merge)
    
   # cv.imshow('watershed',watershed(Skeletonized_Image))
   # numpy_horizontal = np.vstack(((skeletonize(img6//255) * 255),watershed(Skeletonized_Image)))
    bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    horizontal1 = np.hstack((img,img2,GAUSSTHRESH))
    bgrhorizontal1 = cv.cvtColor(horizontal1.astype(np.uint8), cv.COLOR_GRAY2BGR)
    
    
    bgrSkeletonized_Image = cv.cvtColor(Skeletonized_Image.astype(np.uint8), cv.COLOR_GRAY2BGR)
    #bgrMerge = cv.cvtColor(Merge.astype(np.uint8), cv.COLOR_GRAY2BGR)
    bgrimg3 = cv.cvtColor(img3.astype(np.uint8), cv.COLOR_GRAY2BGR)
    horizontal2= np.hstack((bgrimg3,Merge,Watershed))
    
    largewindow=np.vstack((bgrhorizontal1,horizontal2))
    
    newdim=(int(np.shape(largewindow)[0]*0.7),int(np.shape(largewindow)[1]*0.4))
    resizedwindow =cv.resize(largewindow,newdim)
    cv.imshow('skeleton',img)
    image_object.set_data(largewindow)
    myfigure.canvas.draw()
    #plt.clf()
    #plt.imshow(largewindow)
    # cv.imshow('watershed',maxs)

cv.destroyAllWindows()


