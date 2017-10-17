from time import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,discriminant_analysis, random_projection)
import cv2
import os
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN,Birch
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import Counter
import sys

#Define main Directory, which contains the seperate images in different folders
mainDir="E:/DLRC/data/7k/5k/"
folders=os.listdir(mainDir)
images=[]
dict1={}
nextCounter=0
originalImages=[]
for folder in folders:    
    prevCounter=nextCounter
    nextCounter=0
    for image in os.listdir(mainDir+folder+"/"):
        im=cv2.imread(mainDir+folder+"/"+image)
        originalImages.append(im)
        im = list(im.flat)
        images.append(im)
        nextCounter+=1
    nextCounter+=prevCounter    
    dict1[folder]=(prevCounter,nextCounter)

images=np.array(images) 
labels = list(np.repeat(0,500))+list(np.repeat(1,500))+list(np.repeat(2,500))+list(np.repeat(3,500))+list(np.repeat(4,500))

X = images
y = labels

n_samples, n_features = X.shape
n_neighbors = 30

#Define the plotting for the embedding
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(labels[i]),color=plt.cm.Set1(y[i] / 10.),fontdict={'weight': 'bold', 'size': 9})
    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])
        for i in range(images.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                continue
            shown_images = np.r_[shown_images, [X[i]]]
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


# Projection on to the first 2 principal components
print("PCA projection")
t0 = time()
X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
plot_embedding(X_pca,"Principal Components projection of the images (time %.2fs)" %(time() - t0))

# t-SNE embedding
print("t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(X)

plot_embedding(X_tsne[:135],"t-SNE embedding (time %.2fs)" %(time() - t0))

plt.show()