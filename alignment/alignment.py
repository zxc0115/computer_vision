from ImagePyramid import *
import cv2
import numpy as np
import os
def split_pic(img):
    row = img.shape[0] // 3
    return img[2*row:3*row],img[row:2*row],img[0:row]

# choose best row col according to SSD
def ssd(base_image, move_image, searchX=0.25, searchY=0.25, offsetX=0, offsetY=0):
    row,col = base_image.shape
    new_image = np.array(base_image)
    m_ysize = move_image.shape[0]
    m_xsize = move_image.shape[1]
    mhalf_ysize = move_image.shape[0]//2
    mhalf_xsize = move_image.shape[1]//2

    padding_image = np.zeros((row+m_ysize,col+m_xsize))
    padding_image[mhalf_ysize:row+mhalf_ysize,mhalf_xsize:col+mhalf_xsize] = base_image

    best_score = 1e20
    best_row = -1
    best_col = -1
    for i in range(int(row*(0.5-searchY))+offsetY,int(row*(0.5+searchY))+offsetY):
        for j in range(int(col*(0.5-searchX))+offsetX,int(col*(0.5+searchX))+offsetX):
            sub_region = padding_image[i:i+m_ysize,j:j+m_xsize]
            score = np.sum(np.square(sub_region - move_image))
            if score < best_score:
                best_row = i
                best_col = j
                best_score = score
    return best_score,best_row-mhalf_ysize,best_col-mhalf_xsize

# align and map onto large image and 
def channel_match(img1, img2, offsetY2, offsetX2):
    row = img1.shape[0]
    col = img1.shape[1]

    newImage1 = np.array(img1)
    newImage2 = np.zeros_like(img2)

    if offsetY2>0 and offsetX2>0:
        newImage2[offsetY2:,offsetX2:] = img2[:-offsetY2,:-offsetX2]
    elif offsetY2>0 and offsetX2<0:
        newImage2[offsetY2:,:offsetX2] = img2[:-offsetY2,-offsetX2:]
    elif offsetY2<0 and offsetX2>0:
        newImage2[:offsetY2,offsetX2:] = img2[-offsetY2:,:-offsetX2]
    elif offsetY2<0 and offsetX2<0:
        newImage2[:offsetY2,:offsetX2] = img2[-offsetY2:,-offsetX2:]
        
    elif offsetY2<0 and offsetX2==0:
        newImage2[:offsetY2,:] = img2[-offsetY2:,:]
    elif offsetY2>0 and offsetX2==0:
        newImage2[offsetY2:,:] = img2[:-offsetY2,:]
    elif offsetY2==0 and offsetX2<0:
        newImage2[:,:offsetX2] = img2[:,-offsetX2:]
    elif offsetY2==0 and offsetX2>0:
        newImage2[:,offsetX2:] = img2[:,:-offsetX2]
    else:
        newImage2 = np.array(img2)
    return newImage1,newImage2

def coloring_task(filename):
    image = cv2.imread(filename,0)
    image = image/255
    R,G,B = split_pic(image)
    Bd = dePadding(B, 300)
    Gd = dePadding(G, 300)
    Rd = dePadding(R, 300)

    B_l1 = pyramidDown(Bd)
    G_l1 = pyramidDown(Gd)
    R_l1 = pyramidDown(Rd)
    print("level1")
    B_l2 = pyramidDown(B_l1)
    G_l2 = pyramidDown(G_l1)
    R_l2 = pyramidDown(R_l1)
    print("level2")
    B_l3 = pyramidDown(B_l2)
    G_l3 = pyramidDown(G_l2)
    R_l3 = pyramidDown(R_l2)
    print("level3")
    B_l4 = pyramidDown(B_l3)
    G_l4 = pyramidDown(G_l3)
    R_l4 = pyramidDown(R_l3)
    print("level4")
    B_l5 = pyramidDown(B_l4)
    G_l5 = pyramidDown(G_l4)
    R_l5 = pyramidDown(R_l4)
    print("level5")

    cv2.namedWindow("R",cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("G",cv2.WINDOW_AUTOSIZE) 
    cv2.namedWindow("B",cv2.WINDOW_AUTOSIZE) 
    cv2.namedWindow("",cv2.WINDOW_AUTOSIZE) 
    cv2.imshow("R",Rd)
    cv2.imshow("G",Gd)
    cv2.imshow("B",Bd)

    # According to result, change green or blue as base image to get better
    bscore1,brow1,bcol1 = ssd(G_l5, B_l5, searchX=0.4, searchY=0.4)
    print(bscore1,brow1,bcol1)
    bscore2,brow2,bcol2 = ssd(G_l5, R_l5, searchX=0.4, searchY=0.4)
    print(bscore2,brow2,bcol2)
    
    bscore1,brow1,bcol1 = ssd(G_l4, B_l4, searchX=0.3, searchY=0.3, offsetX=bcol1*2, offsetY=brow1*2)
    print(bscore1,brow1,bcol1)
    bscore2,brow2,bcol2 = ssd(G_l4, R_l4, searchX=0.3, searchY=0.3, offsetX=bcol2*2, offsetY=brow2*2)
    print(bscore2,brow2,bcol2)
    
    bscore1,brow1,bcol1 = ssd(G_l3, B_l3, searchX=0.2, searchY=0.2, offsetX=bcol1*2, offsetY=brow1*2)
    print(bscore1,brow1,bcol1)
    bscore2,brow2,bcol2 = ssd(G_l3, R_l3, searchX=0.2, searchY=0.2, offsetX=bcol2*2, offsetY=brow2*2)
    print(bscore2,brow2,bcol2)
    
    bscore1,brow1,bcol1 = ssd(G_l2, B_l2, searchX=0.12, searchY=0.15, offsetX=bcol1*2, offsetY=brow1*2)
    print(bscore1,brow1,bcol1)
    bscore2,brow2,bcol2 = ssd(G_l2, R_l2, searchX=0.12, searchY=0.15, offsetX=bcol2*2, offsetY=brow2*2)
    print(bscore2,brow2,bcol2)

    bscore1,brow1,bcol1 = ssd(G_l1, B_l1, searchX=0.04, searchY=0.08, offsetX=bcol1*2, offsetY=brow1*2)
    print(bscore1,brow1,bcol1)
    bscore2,brow2,bcol2 = ssd(G_l1, R_l1, searchX=0.04, searchY=0.08, offsetX=bcol2*2, offsetY=brow2*2)
    print(bscore2,brow2,bcol2)

    bscore1,brow1,bcol1 = ssd(Gd, Bd, searchX=0.01, searchY=0.01, offsetX=bcol1*2, offsetY=brow1*2)
    print(bscore1,brow1,bcol1)
    bscore2,brow2,bcol2 = ssd(Gd, Rd, searchX=0.01, searchY=0.01, offsetX=bcol2*2, offsetY=brow2*2)
    print(bscore2,brow2,bcol2)
    

    newG, newB = channel_match(G, B, brow1, bcol1)
    newG, newR = channel_match(G, R, brow2, bcol2)
    colorImg = np.dstack((newB, newG, newR))
    cv2.imshow("",colorImg)
    saveFloatImage(filename.split('.')[0]+"_coloring.png",colorImg)
    cv2.waitKey(0)

if __name__ == '__main__':
    filename = ".//data//train.tif"
    coloring_task(filename)
    