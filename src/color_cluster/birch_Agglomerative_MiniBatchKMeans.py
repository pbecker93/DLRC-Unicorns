# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 17:16:31 2017

@author: Shayan
"""

import cv2
import os
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN,Birch,MiniBatchKMeans
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.preprocessing import scale
import math
import copy

#DEFINE NUMBER OF CLUSTERS HERE
n_clusters=4

mainDir="E:/DLRC/data/cropoktold/"
mainDir="E:/DLRC/data/Shayans/"
folders=os.listdir(mainDir)
images=[]
dict1={}
nextCounter=0

#Uncomment for equalizing the histogram
#def equalize_hist(img):
#    for c in range(0, 2):
#       img[:,:,c] = cv2.equalizeHist(img[:,:,c])
#    return img
#
#Uncomment for adaptive equalization of the histogram
# def clahe(img):
#     lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#     cl = clahe.apply(l)
#     limg = cv2.merge((cl,a,b))
#     final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
#     return final


#
#def otsuThresh(path):
#    colored = cv2.imread(path)
#    colored = cv2.resize(colored, (44, 44)) 
#    
#    colorcopy = copy.deepcopy(colored)
#
#    gray = cv2.cvtColor(colored,cv2.COLOR_BGR2GRAY)
#    blur = cv2.medianBlur(gray,5)
#
#    blur = cv2.GaussianBlur(gray,(5,5),0)
#    mask = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
#
#    mask = cv2.adaptiveThreshold(mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#    mask = cv2.adaptiveThreshold(mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#    mask = cv2.adaptiveThreshold(mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#
#    ret3,mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    mask_inv = cv2.bitwise_not(mask)
#    
#    img1_bg = cv2.bitwise_and(colored,colored,mask = mask_inv)
#    img2_fg = cv2.bitwise_and(colorcopy,colorcopy,mask = mask)
#    
#    dst = cv2.add(img1_bg,img2_fg)
#    
#    cv2.imshow("md",img2_fg)
#    cv2.waitKey(0)
#    return dst

#Preprocessing for images, Morphological opening and closing, uncomment code to apply transformations
#and merge all the features
def preprocessing(im):
    im = cv2.resize(im, (44, 44)) 
    kernel = np.ones((5,5),np.uint8)
#    im = cv2.medianBlur(im,5)
#    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    im = cv2.medianBlur(im,5)
#    im = cv2.medianBlur(im,5)
#    im=clahe(im)#cv2.GaussianBlur(clahe(cv2.GaussianBlur(clahe(im),(5,5),0)),(5,5),0)
#    rows,cols = im.shape[0],im.shape[1]
#    im = cv2.blur(im,(5,5))
#    M1 = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
#    dst1 = cv2.warpAffine(im,M1,(cols,rows))
#    dst1 = list(dst1.flat)    
#    M = np.float32([[1,0,100],[0,1,50]])
#    dst = cv2.warpAffine(im,M,(cols,rows))
#    dst = list(dst.flat)    
    im = list(im.flat)
    return im
#    return im+dst1+dst
    

#read all images in the main directory    
for folder in folders:    
    prevCounter=nextCounter
    nextCounter=0
    for image in os.listdir(mainDir+folder+"/"):
        path=mainDir+folder+"/"+image
        im = cv2.imread(path)

        images.append(preprocessing(im))
        nextCounter+=1
    nextCounter+=prevCounter    
    dict1[folder]=(prevCounter,nextCounter)
    
images=np.array(images) 
images = scale(images)

X = StandardScaler().fit_transform(images)


#perform agglomerative clustering, Birch and MiniBatchKMeans clustering
ac=AgglomerativeClustering(linkage="complete", affinity="l1",n_clusters=n_clusters).fit_predict(X)

originalLabels = copy.deepcopy(ac)

for key in dict1.keys():
   x,y=dict1[key]
   print("Key--->",key)
   print(Counter(ac[x:y]).most_common())
   originalLabels[x:y]=Counter(ac[x:y]).most_common()[0][0]

print ("-"*80)

ac = Birch(n_clusters=n_clusters,compute_labels=True).fit_predict(X)
originalLabels = copy.deepcopy(ac)

for key in dict1.keys():
    x,y=dict1[key]
    print("Key--->",key)
    print(Counter(ac[x:y]).most_common())
    originalLabels[x:y]=Counter(ac[x:y]).most_common()[0][0]
    
# Compute clustering with MiniBatchKMeans.
print ("-"*80)

mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=770,
                      n_init=100, max_no_improvement=None, verbose=0,
                      random_state=0,compute_labels=True).fit_predict(X)

originalLabels = copy.deepcopy(mbk)

for key in dict1.keys():
    x,y=dict1[key]
    print("Key--->",key)
    print(Counter(mbk[x:y]).most_common())
    originalLabels[x:y]=Counter(mbk[x:y]).most_common()[0][0]


                      
#Train two classifiers on the ouput of any of the clusterig methods above.    
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
clf = tree.DecisionTreeClassifier()

ac=list(ac)
for n,i in enumerate(ac):
   if i==3:
       ac[n]="red"
   elif i==1:
       ac[n]="white"
   elif i==0:
       ac[n]="blue"
   elif i==2:
       ac[n]="yellow"       

clf = clf.fit(images, ac)
neigh = KNeighborsClassifier(n_neighbors=10000)
neigh.fit(images, ac)
print(neigh.predict([[1.1]]))    

from sklearn.externals import joblib
joblib.dump(neigh, 'JAC.pkl') 
test=cv2.imread("E:/DLRC/data/test/4.jpg")
print(neigh.predict(np.array(preprocessing(test)).reshape(1,-1)))