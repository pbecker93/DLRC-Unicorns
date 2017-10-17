import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import cv2

import time
start=time.time()
X = cv2.imread('1507121681.1548116900.jpg')

kernel = np.ones((5,5),np.uint8)
X = cv2.erode(X,kernel,iterations = 10)
X = cv2.morphologyEx(X, cv2.MORPH_OPEN, kernel)
X = cv2.morphologyEx(X, cv2.MORPH_CLOSE, kernel)
X = cv2.blur(X,(5,5))
(a,b)=(64,64)
X = cv2.resize(X,(a,b))
#cv2.imshow("xsmall",x)
#cv2.waitKey(0)

X = X.reshape((-1,3))
original=X.reshape(a,b,3)
cv2.imshow("original",original)

X = StandardScaler().fit_transform(X)

db = DBSCAN().fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)


import copy
clusters=[]
for i in range(0,n_clusters_):
    temp=copy.deepcopy(X)
    temp[labels==i]=1
    temp[labels!=i]=0
    clusters.append(temp)



for i in range(0,len(clusters)): 
    cluster=clusters[i]    
    cluster=cluster.reshape(a,b,3)
    cv2.imshow(str(i),cluster)

cv2.waitKey(0)