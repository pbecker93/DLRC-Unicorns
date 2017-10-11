import numpy as np
import cv2
import chili_tag_detector as ctd
from color_cluster.siamese_cluster import SiameseClustering
#from color_cluster.mean_cluster import ColorCluster
import time
class Detector:

    def __init__(self, model):
        self.model = model
        #self.color_cluster= ColorCluster()
        self.color_cluster = SiameseClustering("/home/nvidia/RobotControl/src/color_cluster/model/", 2, 4)

        mask = np.zeros((480, 640), dtype=np.uint8)
        self.mask = cv2.fillConvexPoly(mask, np.array([[275, 285], [365, 285], [430, 166], [210, 166]]), 1, 1)

    def detect_brick_gripper(self, image):
        heat_map = self.model.extract_hmap(image)[0] * self.mask 
        #cv2.imshow("camera_down", image)
        cv2.imshow("heatmap_top " , heat_map)

        point = self._get_center_of_largest_contour(heat_map)
        if point is not None and point[1] > 220:
            return self.color_cluster.run_cluster(point, image)
        else:
            return None


    def _get_center_of_largest_contour(self, heat_map):
  #      gray = cv2.cvtColor(heat_map, cv2.COLOR_BGR2GRAY)
        _, contours, _ = cv2.findContours(heat_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print(len(contours))
        if len(contours) > 0:
            largest_contour = contours[0]
            if np.array(largest_contour.reshape(-1, 2).shape[0]) > 2:
                return np.mean(np.array(largest_contour).reshape(-1, 2), axis=0).astype(np.int)
        return None

# Todo Delete if not used!!!
#    def detect_brick(self, img):
     #   print(img.shape)
#        lego_map, obs_map = self.model.extract_hmap(img)
     #   cv2.imshow("heat_map", h_map)
#        closest_obstacle = self.avoider.getObstacle(obs_map)
#        if closest_obstacle is not None:
#            return closest_obstacle
#        else:
#            return self._get_center_of_largest_contour(lego_map)

    @staticmethod
    def _measure_squareness(points):
        d1 = np.linalg.norm(points[0] - points[1])
        d2 = np.linalg.norm(points[1] - points[2])
        return d1 / d2

    @staticmethod
    def detect_box(img, box_id):
#        print(img.shape)
        marker_tuple = ctd.detect(img)
#        print(marker_tuple)
        best_marker = None
        best_value = 0
        for t in marker_tuple:
            if t[0] == box_id:
                squareness = Detector._measure_squareness(t[1])
                #print(squareness)
                if squareness > best_value:
                    best_marker = np.mean(t[1], 0).astype(np.int32)
                    best_value = squareness
       # print(best_marker)
        return best_marker

