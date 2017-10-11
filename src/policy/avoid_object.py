import numpy as np
import cv2

class AvoidObject:
    THRESH_Y = 380
    THRESH_X = 320
    BASELINE = 50
    FACTOR = 0.3

    def _scan_contour(self, contour):
        
        max_y = np.max(contour, axis=0).astype(np.int)[1]
        center = np.mean(contour, axis=0).astype(int)
        return [center, max_y]

    def getObstacle(self, obstacle_map):

        _, contours,_ = cv2.findContours(obstacle_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        obstacles = list()
        if len(contours) != 0:
            for i in range(len(contours)):
                cnt = np.array(contours[i]).reshape(-1, 2)

                if cnt.shape[0] >200:
                    obstacles.append(self._scan_contour(cnt))
        # If there are objects detected we loop through them and and check if they are close enough
        if len(obstacles)!=0:
            for obs in obstacles:
                # Check if the detected object is in the Polygon mask and if it is close enough         
                if AvoidObject.THRESH_Y - 80 < obs[1] < AvoidObject.THRESH_Y \
                        and AvoidObject.THRESH_X - 200 < obs[0][0] < AvoidObject.THRESH_X + 200:
                    obstacle_map = cv2.drawMarker(obstacle_map, tuple(obs[0]), (255,0,0) ,10)
                    cv2.imshow("O_map",obstacle_map)
                    return obs
        cv2.imshow("O_map",obstacle_map)
        return None
