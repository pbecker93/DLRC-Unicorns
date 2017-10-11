import cv2
import numpy as np
from sklearn.externals import joblib
import time
class ColorCluster:
    @staticmethod
    def _clahe(img):
        lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final

    def __init__(self):
        self.cluster = {}
        self.clf = joblib.load('JAC.pkl')
        print("shay clust")
 
    def _crop(self, point, img):
        point[1] += 15
        size = 22
        x1 = point[0] - size
        x2 = point[0] + size
        y1 = point[1] - size
        y2 = point[1] + size

        return img[y1:y2, x1:x2, :]

    def run_cluster(self, point, img):

        img = self._crop(point, img)
        cv2.imwrite(str(time.time())+'crop.jpg',img)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        cv2.imshow("Crop ",img)

        #img = cv2.resize(img, (44, 44))           
        #img=cv2.GaussianBlur(self._clahe(cv2.GaussianBlur(self._clahe(img),(5,5),0)),(5,5),0)
        #cv2.imshow("crop2", img) 
        #img =np.array(list(img.flat))
        #print(img.shape)
        #print(self.clf.predict(img.reshape(1,-1)))
        return 0

