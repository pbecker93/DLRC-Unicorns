from color_cluster.siamese_model import Model
import numpy as np
import cv2
import time
class SiameseClustering:


    def __init__(self, model_path, threshold, max_clusters):

        self.m = Model((64, 64))
        self.m.load_model(model_path)
        self.threshold = threshold
        self.max_clusters = max_clusters
        self.ct = 0
        self.cluster_dict = {}

    def run_cluster(self, point, img):
        patch = self._crop(point, img)
        cv2.imwrite(str(time.time()) +'.jpg' , patch)
        return self._cluster(patch)

    def _cluster(self, img_patch):

        latent = self.m.get_latent(np.expand_dims(img_patch, 0))
        print(latent)
        if len(self.cluster_dict.keys()) == 0:
            self.cluster_dict[0] = latent
            return 0
        else:
            means = self._get_cluster_means()
            diffs = np.linalg.norm(means - latent, axis=-1)
            # Greater than threshold and still place
            if np.min(diffs) > self.threshold and len(self.cluster_dict.keys()) < self.max_clusters:
                cluster = len(self.cluster_dict.keys())
                self.cluster_dict[cluster] = latent
                return cluster
            # near to existing or full
            else:
                cluster = np.argmin(diffs)
                self.cluster_dict[cluster] = np.concatenate((self.cluster_dict[cluster], latent), 0)
                return cluster

    def _get_cluster_means(self):
        mean_list = list()
        for i in self.cluster_dict.keys():
            mean = np.mean(self.cluster_dict[i], 0)
#            print(mean)
            mean_list.append(mean)
        return np.stack(mean_list)

    def _crop(self, point, img):
        size = 32 
        point[1] +=18
        x1 = point[0] - size
        x2 = point[0] + size
        y1 = point[1] - size
        y2 = point[1] + size

        return img[y1:y2, x1:x2, :]

