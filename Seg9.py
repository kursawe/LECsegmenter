#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:23:53 2019

@author: Thibaut Goldsborough, tg76@st-andrews.ac.uk
"""
#basepath ="./STACKS/SPJob1_120218_1"
#basepath ="./STACKS/STACKS_0"
basepath ="./STACKS"
#%matplotlib auto
import os, sys
import numpy as np
import cv2 as cv
from skimage.morphology import skeletonize
from random import randrange
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
import pickle as pickle
import easygui as gui
from xlwt import Workbook 

global outcome, CELL_DICTIONARY, MAX_NUMBER_OF_CELLS, tiff_images, dim1, dim2
MAX_NUMBER_OF_CELLS=0
  
answer=gui.buttonbox("Would you like to Start Over or Continue Previous Project?",choices=("Start Over","Previous Project"))

if answer=="Start Over":
    answer=True
if answer=="Previous Project":
    answer=False

if answer==False:  
    with open('SAVED_WORK', 'rb') as infile:
        outcome = pickle.load(infile) 

def merges_red(img1,img2,amount):
    overlay = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    b,g,r = cv.split(overlay)
    r = cv.add(r,amount, dst = r, mask =img2, dtype = cv.CV_8U)
    merged=cv.merge((b,g,r),img1)
    return merged
def merges_blue(img1,img2,amount):
    b,g,r = cv.split(img1)
    b = cv.add(b,amount, dst = b, mask =img2, dtype = cv.CV_8U)
    merged=cv.merge((b,g,r),img1)
    return merged
def merges_green(img1,img2,amount):
    b,g,r = cv.split(img1)
    g = cv.add(g,amount, dst = g, mask =img2, dtype = cv.CV_8U)
    merged=cv.merge((b,g,r),img1)
    return merged

def removesmallelements(img,minsize):
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
    global b1,stats
    edges= np.pad(np.ones((dim1-2,dim2-2)), pad_width=1, mode='constant', constant_values=0)    
    img=img*edges
    img=img.astype(np.uint8)
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(cv.bitwise_not(img),\
    connectivity=4)
    b1=np.zeros((dim1,dim2)).astype(np.uint8)
    g1=np.zeros((dim1,dim2)).astype(np.uint8)
    r1=np.zeros((dim1,dim2)).astype(np.uint8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1   
    for i in range(0, nb_components):
        if sizes[i] <=10 :
            output[output==i+1]=0 
    output=cv.dilate(output.astype(np.uint8),None,iterations=1)
    for i in range(0, nb_components):     
        if sizes[i] <=10000 : #MAX CELL SIZE     
            b1[output == i + 1]=colors[i][0]
            g1[output == i + 1]=colors[i][1]
            r1[output == i + 1]=colors[i][2]
    image=cv.merge((b1,g1,r1))  
    return (image,output)

def simple_watershed(img):
    global output,Centroid_list,nb_components
    Centroid_list=[]
    edges= np.pad(np.ones((dim1-2,dim2-2)), pad_width=1, mode='constant', constant_values=0)   
   # print(img,edges)
    img=img*edges
    
    img=img.astype(np.uint8)
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(cv.bitwise_not(img),\
    connectivity=4)
    sizes = stats[1:, -1]; nb_components = nb_components -1  
    for i in range(0, nb_components):
        if sizes[i] <=10 :
            output[output==i+1]=0 
    for cell_index in range(0,nb_components+1):
        if cell_index>1:
            if np.where(output==cell_index)[0].size>0:
                Centroid_list.append(((np.mean(np.where(output==cell_index)[0]),np.mean(np.where(output==cell_index)[1])),cell_index))     
    return (output,Centroid_list)


def get_centroids(frame):
    Centroid_list=[]
    nb_components=list(np.unique(frame))
    for cell_index in nb_components:
        if cell_index>1:
            if np.where(frame==cell_index)[0].size>0:
                Centroid_list.append(((np.mean(np.where(frame==cell_index)[0]),np.mean(np.where(frame==cell_index)[1])),cell_index))
    return (Centroid_list)

def get_stats(input_data):
    global cnt,rect, CELL_DICTIONARY
    background=np.zeros((dim1,dim2))
    CELL_DICTIONARY={} 
    nb_components=list(np.unique(input_data[0][1]))      
    for cell_index in range(MAX_NUMBER_OF_CELLS+1):
        if cell_index>1:
            CELL_DICTIONARY[cell_index]=[]

    for frame_index in range(len(input_data)): 
        frame=input_data[frame_index][1].copy()
        nb_components=list(np.unique(frame))
        for cell_index in nb_components:
            if cell_index>1:
                if np.where(frame==cell_index)[0].size>0:
                    background[frame==cell_index]=255
                    contours,hierarchy = cv.findContours(background.astype(np.uint8), 1,method= cv.CHAIN_APPROX_NONE)
                    cnt = contours[0]
                    cy,cx=int(np.mean(np.where(frame==cell_index)[0])),int(np.mean(np.where(frame==cell_index)[1]))
                    area = cv.contourArea(cnt)
                    perimeter = cv.arcLength(cnt,True)
                    rect = cv.minAreaRect(cnt)
                    width=min(rect[1][1],rect[1][0])
                    length=max(rect[1][1],rect[1][0])
                    angle=rect[2]+90
                    if rect[1][1] > rect[1][0]:
                        angle=rect[2]    
                    #if len(cnt)>5:
                      #  (x,y),(MA,ma),angle = cv.fitEllipse(cnt)
                    vert_height = cv.boundingRect(cnt)[3]
                    hoz_width=len(list(np.where(np.where(frame==cell_index)[1]==cx)[0]))
                    CELL_DICTIONARY[cell_index].append(((cx,cy),int(area),int(perimeter),int(width),int(length),int(angle),vert_height,hoz_width))                              
                    background=np.zeros((dim1,dim2))
        for cell_index in range(MAX_NUMBER_OF_CELLS+1):
            if cell_index>1:
                if len(CELL_DICTIONARY[cell_index])<=frame_index:
                    CELL_DICTIONARY[cell_index].append((("NA","NA"),"NA","NA","NA","NA","NA","NA","NA"))
                    
    for cell_index in range(MAX_NUMBER_OF_CELLS+1):
        if cell_index>1:
            if all(elem ==(('NA', 'NA'), 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA') for elem in CELL_DICTIONARY[cell_index])==True:
                del(CELL_DICTIONARY[cell_index])
                
        
    return (CELL_DICTIONARY)

def follow_cells_and_watershed(prev_img,img):
    #prev_img has to be mask and img has to be membrane contours
    global pairs_of_cells,mask,MAX_NUMBER_OF_CELLS
    centroids_prev=get_centroids(prev_img)
    post_output,centroids_post=simple_watershed(img)
    pairs_of_cells=[]
    for centroid1num in centroids_prev:
        dist_list=[]
        centroid1=centroid1num[0]
        for centroid2num in centroids_post:
            centroid2=centroid2num[0]
            dist_list.append((np.sqrt((centroid1[0]-centroid2[0])**2+((centroid1[1]-centroid2[1])**2)),(centroid1num[1],centroid2num[1])))
        if min(dist_list, key = lambda t: t[0])[0]<50:
            pairs_of_cells.append((min(dist_list, key = lambda t: t[0])[1],min(dist_list, key = lambda t: t[0])[0]))
    mask=np.zeros((dim1,dim2))
    
    post_cell_nums = [lis[0][1] for lis in pairs_of_cells]
    pre_cell_nums= [lis[0][0] for lis in pairs_of_cells]
        
    MAX_NUMBER_OF_CELLS=max(MAX_NUMBER_OF_CELLS,max(post_cell_nums),max(pre_cell_nums))
    
    for cell_index in np.unique(post_output):
        if cell_index>1:
            if (cell_index in post_cell_nums)==False:
                MAX_NUMBER_OF_CELLS+=1
                pairs_of_cells.append(((MAX_NUMBER_OF_CELLS,cell_index),1000))
         
    cell_num=len(pairs_of_cells)
    for pair_index in range(cell_num):
        pair=pairs_of_cells[pair_index]
        for pair2_index in range(cell_num):
            pair2=pairs_of_cells[pair2_index]
            if pair2[0][1]==pair[0][1]:
                if pair2[0][0]!=pair[0][0]:
                    pairs_of_cells[pair_index]=pairs_of_cells[pair2_index]=min((pair,pair2), key = lambda t: t[1])
        #mask[post_output==pair[0][1]]=pair[0][0]
    
    for pair in pairs_of_cells:                                         
        mask[post_output==pair[0][1]]=pair[0][0]
            
    mask=cv.dilate( mask.astype(np.uint8),None,iterations=1)
    
    b2=np.zeros((dim1,dim2)).astype(np.uint8)
    g2=np.zeros((dim1,dim2)).astype(np.uint8)
    r2=np.zeros((dim1,dim2)).astype(np.uint8)
    for i in (np.unique( mask).astype(np.uint8)): 
        if i>1:
            b2[ mask == i ]=colors[i-1][0]
            g2[ mask == i ]=colors[i-1][1]
            r2[ mask == i  ]=colors[i-1][2]
    image=cv.merge((b2,g2,r2))   
    return image, mask

def GUI(event,x,y,flags,param):
    global Skeletonized_Image,Cursor
    global drawing, X,Y ,been
    global saved_list,dim, iter_photo
    
    Cursor=np.zeros((dim1,dim2)).astype(np.uint8)
    Skeletonized_Image=saved_list[len(saved_list)-1].copy()
    if drawing==True:
        if mode==True:
            if event == cv.EVENT_LBUTTONDOWN:
               debug = cv.line(Skeletonized_Image,(x,y),(X,Y),(255),1).copy()
               saved_list.append(debug)
               update_numbers(saved_list[-1],iter_photo)
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
        cv.circle(Cursor,(x,y),dim, (1), 0)
        cv.circle(Skeletonized_Image,(x,y),dim, (0), -1)
        if event== cv.EVENT_LBUTTONDOWN:            
            saved_list.append(Skeletonized_Image.copy())
            update_numbers(saved_list[-1],iter_photo)

def process_image(img,a,b,c,d):
    if c <1:
        c=1
    BLURED= cv.GaussianBlur(img,(5,5),0)
    GAUSSTHRESH=cv.adaptiveThreshold(BLURED,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,2*c+1,0)  
    rem=removesmallelements(GAUSSTHRESH,1000)
    img3 = cv.GaussianBlur(rem,(5,5),0)
    ret,img4 = cv.threshold(img3,a,255,cv.THRESH_BINARY)
    rem=removesmallelements(img4,1000)
    img5 = cv.GaussianBlur(rem,(5,5),0)
    ret,img6 = cv.threshold(img5,b,255,cv.THRESH_BINARY)
    Skeletonized_Image = (skeletonize(img6//255) * 255).astype(np.uint8)
    Watershed=watershed(Skeletonized_Image)[0]  
    img6=cv.cvtColor(img6.astype(np.uint8), cv.COLOR_GRAY2BGR)
    Skeletonized_Image_BGR=merges_red(img//2,Skeletonized_Image,255)
    return (img,GAUSSTHRESH,img4.astype(np.uint8),img6,Skeletonized_Image_BGR,Watershed,Skeletonized_Image)

def update_numbers(membrane_outlines,frame_num,speed="Fast"):
    global Numbers, tiff_images
    Image=tiff_images[frame_num]
    Base_Image=2*(Image[0]+Image[1])
    
    Numbers=np.zeros((dim1,dim2)).astype(np.uint8)
    if speed=="Slow":  
        mask=outcome[frame_num][1].copy()
        centroids=get_centroids(mask)
        
    if speed=="Fast":
        mask=watershed(membrane_outlines)[1]
        centroids=get_centroids(mask)
               
    for centroid in centroids:
        cv.putText(Numbers,str(centroid[1]),org=(int(centroid[0][1]-5),int(centroid[0][0]+5)),fontFace=cv.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.3,thickness=0,color=(1))

    pre_left_panel=merges_red(Base_Image//2,membrane_outlines,255)
    left_panel=merges_blue(pre_left_panel,Numbers,255)

    return left_panel

def save_all_work(boolean):
      for frame_num in range(len(outcome)):
                   
          outline=outcome[frame_num][2].copy()    
          if len(outcome[frame_num][2])==0:
              break
          if frame_num==0:
                Channel3_Mask,Channel1_Mask=watershed(outline)
                outcome[frame_num][1]=Channel1_Mask.copy()
                outcome[frame_num][0]=Channel3_Mask.copy()
                Numbers_Mask=update_numbers(outline,frame_num,"Slow")
                outcome[frame_num][3]=Numbers_Mask.copy()              
          if frame_num>0:
                prev_frame=outcome[frame_num-1][1].copy()
                Channel3_Mask,Channel1_Mask=follow_cells_and_watershed(prev_frame,outline)
                outcome[frame_num][1]=Channel1_Mask.copy()
                outcome[frame_num][0]=Channel3_Mask.copy()
                Numbers_Mask=update_numbers(outline,frame_num,"Slow")
                outcome[frame_num][3]=Numbers_Mask.copy()
          if boolean==True:  
                with open('SAVED_WORK', 'wb') as outfile:
                    pickle.dump(outcome, outfile, pickle.HIGHEST_PROTOCOL)
            
def display(outcome):
    cv.namedWindow('Press E to Exit')  
    frame_num=0  
    while(1):
        k = cv.waitKey(1) & 0xFF
        display_right=outcome[frame_num][0]
        display_left=outcome[frame_num][3]
        display= np.hstack((display_left,display_right))
        cv.imshow('Press E to Exit',display)
        cv.moveWindow('Press E to Exit',150,10)
        if k ==ord('p'):
            if frame_num<(len(outcome)-1):
                frame_num+=1
        if k==ord('o'):
            if frame_num>0:
                frame_num-=1              
        if k==ord('e'):
            break              
        
    cv.destroyAllWindows()
    
def save_excel(outcome,Save_As):
    wb = Workbook() 
    sheet1 = wb.add_sheet('Sheet 1')  
    CELL_DICTIONARY=get_stats(outcome)
    number_of_frames=len(tiff_images)
    sheet1.write(0,0,'Cell Number')
    sheet1.write(0,1,'Parameter')
    row=1
    for i in range(number_of_frames):
        sheet1.write(0,2+i,'Frame'+str(i))

    row=-8
    cell_index2=0
    for cell_index in list(CELL_DICTIONARY.keys()):
        row+=8
        sheet1.write(1+cell_index2*8,1,"Centroid")
        sheet1.write(2+cell_index2*8,1,"Surface Area")
        sheet1.write(3+cell_index2*8,1,"Perimeter")
        sheet1.write(4+cell_index2*8,1,"Width")
        sheet1.write(5+cell_index2*8,1,"Length")
        sheet1.write(6+cell_index2*8,1,"Angle")
        sheet1.write(7+cell_index2*8,1,"Vertical Height")
        sheet1.write(8+cell_index2*8,1,"Horizontal Width")
        cell_index2+=1
        for j in range(8):        
            sheet1.write((j+1)+row,0,str(cell_index))
        for frame in range(number_of_frames):

            sheet1.write(1+row,2+frame,str((CELL_DICTIONARY[cell_index][frame][0])))
            sheet1.write(2+row,2+frame,str((CELL_DICTIONARY[cell_index][frame][1])))
            sheet1.write(3+row,2+frame,str((CELL_DICTIONARY[cell_index][frame][2])))
            sheet1.write(4+row,2+frame,str((CELL_DICTIONARY[cell_index][frame][3])))
            sheet1.write(5+row,2+frame,str((CELL_DICTIONARY[cell_index][frame][4])))
            sheet1.write(6+row,2+frame,str((CELL_DICTIONARY[cell_index][frame][5])))
            sheet1.write(7+row,2+frame,str((CELL_DICTIONARY[cell_index][frame][6])))
            sheet1.write(8+row,2+frame,str((CELL_DICTIONARY[cell_index][frame][7])))
            
            
    wb.save(Save_As)
    
    return CELL_DICTIONARY

def plot_cell_movement(CELL_DICTIONARY):     
    X = np.array(())
    Y= np.array(())
    U = np.array(())
    V = np.array(())
    for cell_index in list(CELL_DICTIONARY.keys()):
        if type(CELL_DICTIONARY[cell_index][-1][0][1])!=str:
            if type(CELL_DICTIONARY[cell_index][0][0][1])!=str:        
                startx=CELL_DICTIONARY[cell_index][0][0][1]
                starty=CELL_DICTIONARY[cell_index][0][0][0]
                X=np.append(X,startx)
                Y=np.append(Y,starty)        
                endx=CELL_DICTIONARY[cell_index][-1][0][1]
                endy=CELL_DICTIONARY[cell_index][-1][0][0]
                vectX=(endx-startx)
                vectY=(endy-starty)    
                U=np.append(U,vectX)
                V=np.append(V,vectY)  
    fig, ax = plt.subplots()
    ax.quiver(X, Y,U,V,units='xy' ,scale=0.2,headwidth=2)   
    plt.grid()    
    ax.set_aspect('equal')  
    plt.xlim(0,dim1)
    plt.ylim(0,dim2)  
    plt.show()

  
    
print("Reading files...")


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

dim1=np.shape(tiff_photo)[0] 
dim2=np.shape(tiff_photo)[1] 
     
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
    tiff_images.append((cv.imread(basepath+"/"+photos[Image],cv.IMREAD_GRAYSCALE),(cv.imread(basepath+"/"+photos[Image-1],cv.IMREAD_GRAYSCALE))))
 
    

Info_Sheet=cv.imread("./Info_Sheet.tiff",cv.IMREAD_GRAYSCALE)
Info_str="""

                                Info Sheet
        
 Commands:   
    "p" --> Go to next frame
    "o" --> Go back one frame
    "l" --> Switch between Manual segmentation and Automatic segmentation
    "s" --> Save
    
 User Interface:   
    "m" --> Switch between Drawing/Erasing
    "z" --> Undo
    
    Drawing:   
        Click at two points to draw a line segment
    
    Erasing:    
        "b" --> Increase diameter of eraser
        "n" --> Decrease diameter of eraser
        

    If you have any questions or comments please email tg76@st-andrews.ac.uk

"""

colors=[]
for i in range(10000):
    colors.append((randrange(255),(randrange(255)),(randrange(255))))


iter_photo=0
Display_Mode=False


if answer==True:
    outcome=[]
    for i in range(len(tiff_images)):
        outcome.append([])
        for j in range(4):
            outcome[i].append([])


dim=1 
saved_once2=False
while iter_photo < len(tiff_images):

    Cursor=np.zeros((dim1,dim2)).astype(np.uint8)
    Numbers=np.zeros((dim1,dim2)).astype(np.uint8)
    Image=tiff_images[iter_photo]
    Image_str='Image'+str(iter_photo+1) 
    #img=4*np.maximum(Image[0],Image[1])
    Base_Image=2*(Image[0]+Image[1])
    cv.destroyAllWindows()
    Image_str="GUI "+Image_str
    lag_iter=iter_photo
    mode = True  
    drawing=False 
    been=False
    
    saved_list=[]
    saved_list.append(1)
    
    if answer==False:
        Skeletonized_Image=outcome[iter_photo][2]
          
    if answer==True:
        Skeletonized_Image=frame=process_image(Base_Image,118,53,5,1)[6]
        
    saved_list.append(Skeletonized_Image)
    
    cv.namedWindow("Info Sheet",flags=cv.WINDOW_NORMAL)
    cv.moveWindow("Info Sheet",500,10)
    cv.resizeWindow("Info Sheet",600,500) 
    
    if Display_Mode==False:      
        cv.namedWindow(Image_str,flags=cv.WINDOW_NORMAL)
        cv.moveWindow(Image_str,10,10)
        cv.resizeWindow(Image_str,1300,800) 
        switch = 'Contours: 0 : OFF \n1 : ON'
        switch2 = 'Labels: 0 : OFF \n1 : ON'
        cv.createTrackbar(switch, Image_str,1,1,nothing)
        cv.createTrackbar(switch2, Image_str,1,1,nothing)
        cv.setMouseCallback(Image_str,GUI)
    
    if Display_Mode==True: 
        cv.namedWindow(Image_str,flags=cv.WINDOW_NORMAL)
        cv.moveWindow(Image_str,10,10)
        cv.resizeWindow(Image_str,1300,800) 
        cv.createTrackbar('Second Threshold',Image_str,118,200,nothing)
        cv.createTrackbar('Third Threshold',Image_str,53,200,nothing)
        cv.createTrackbar('First (Adaptive) Threshold',Image_str,5,20,nothing)
        cv.createTrackbar('On/Off',Image_str,1,1,nothing)
     
    update_numbers(saved_list[-1],iter_photo)

    while(1):
        a1 = cv.getTrackbarPos(switch,Image_str)
        a2 = cv.getTrackbarPos(switch2,Image_str)
        k = cv.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        if k == ord('p'):
            lag_iter=iter_photo
            iter_photo+=1           
            break 
        if k == ord('o'):
            if iter_photo>0:
                lag_iter=iter_photo
                iter_photo-=1
                break 
        if k == ord('b'):
            dim+=5
        if k == ord('n'):
            if dim>=5:
                dim-=5
            else:
                dim=1
        if k== ord('z'):
           if len(saved_list)>2:         
               del saved_list[-1]
               update_numbers(saved_list[-1],iter_photo)
               
        if k==ord('l'):
            Display_Mode=not Display_Mode
            break
        if k==ord('s'):
            watershed_Image=watershed(saved_list[-1])[0]
            outcome[iter_photo]=(list((saved_list[-1],watershed_Image,saved_list[-1],update_numbers(saved_list[-1],iter_photo))))
            save_all_work(True)
            
        
        if Display_Mode==False:
            Watershed=watershed(saved_list[-1])
            right_panel=Watershed[0] 
            pre_pre_left_panel=merges_red(Base_Image//2,saved_list[-1],a1*255)
            pre_left_panel=merges_blue(pre_pre_left_panel,Numbers,a2*255)
            left_panel=merges_green(pre_left_panel,Cursor,255)
            window= np.hstack((left_panel,right_panel))
            if saved_once2==True:
                saved_once2=False
                saved_list.append(frames[6])
                update_numbers(saved_list[-1],iter_photo)

        if Display_Mode==True: 

            a2 = cv.getTrackbarPos('Second Threshold',Image_str) 
            b2 = cv.getTrackbarPos('Third Threshold',Image_str) #53
            c2= cv.getTrackbarPos('First (Adaptive) Threshold',Image_str)
            d2 = cv.getTrackbarPos('On/Off',Image_str)
            
            frames=process_image(Base_Image,a2,b2,c2,d2)
            horizontal1 = np.hstack((frames[0],frames[1],frames[2]))
            bgrhorizontal1 = cv.cvtColor(horizontal1.astype(np.uint8), cv.COLOR_GRAY2BGR)   
            bgrhorizontal2 = np.hstack((frames[3],frames[4],frames[5]))
            window=np.vstack((bgrhorizontal1,bgrhorizontal2))
            
            saved_once2=True
        
        cv.imshow(Image_str,window)
        cv.imshow("Info Sheet",Info_Sheet) 
    cv.destroyAllWindows()
    watershed_Image=watershed(saved_list[-1])[0]
    outcome[lag_iter]=(list((saved_list[-1],watershed_Image,saved_list[-1],update_numbers(saved_list[-1],lag_iter))))
   
print("Processing....")

answer=gui.buttonbox("Would you like to save this?",choices=("Save","Don't Save"))

if answer=="Save":
    answer=True
if answer=="Don't Save":
    answer=False

#sys.exit(0)

save_all_work(answer)

display(outcome)

CELL_DICTIONARY=save_excel(outcome,'Cell_Info.xls')

print("Done")

#To obtain a matplotlib plot of the cell movements
#plot_cell_movement(CELL_DICTIONARY)
 



