import cv2
import numpy as np

class ColorCluster:

    def __init__(self):
        self.cluster = {}
        print("mean clust") 
    def _crop(self, point, img):
        size = 24
        point[1] +=18
        x1 = point[0] - size
        x2 = point[0] + size
        y1 = point[1] - size
        y2 = point[1] + size

        return img[y1:y2, x1:x2, :]

    def run_cluster(self, point, img):

        img = self._crop(point, img)
        kernel = np.ones((3, 3), np.float32) / 9
        img = cv2.filter2D(img, -1, kernel)
        vals = np.array([np.mean(img[:,:,0]),np.mean(img[:,:,1]), np.mean(img[:,:,2])])
        print(vals)
        cv2.imshow("crop", img)

        L1 = []

        if not self.cluster:
            print("Cluster empty")
            self.cluster[0] = vals
            return 0
        else:
            for key, item in self.cluster.items():

                L1.append(np.linalg.norm(vals-item))
            print("l1", L1)
            if min(L1)<60:
                cluster_index = L1.index(min(L1))
                print("Inserted in cluster: ", L1.index(min(L1)))

            else:
                cluster_index = len(self.cluster)
                print("Created cluster: ", len(self.cluster))
                self.cluster[cluster_index] = vals
            print("Cluster ID", cluster_index)
            return cluster_index

