import cv2
import numpy as np

class Tracker:

    def __init__(self, model, hist_length=2):
        self.model = model
        self.marker = None
        self.hist = list()
        self.hist_len = hist_length

    def _get_markers(self, lego_map):
        _, contours, _ = cv2.findContours(lego_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        markers = []
        if len(contours) != 0:
            for i, cnt in enumerate(contours):
                if np.array(cnt).reshape(-1, 2).shape[0] > 5:
                    point = np.mean(np.array(cnt).reshape(-1, 2), axis=0).astype(int)
                    markers.append([point[0], point[1]])

        return markers

    def track(self, lego_map):
        if self.marker is None:
            markers = self._get_markers(lego_map)
            print(markers)
            if len(markers) > 0:
                marker = sorted(markers, key=lambda x: x[1])[-1]
                self.marker = np.array(marker)
                return self.marker
            else:
                return None
        else:
            markers = np.array(self._get_markers(lego_map))
            if len(markers) > 0:
                dist = np.linalg.norm(markers-self.marker,axis=1)
                _marker = markers[dist.argmin()]
                self.marker = _marker
                return self.marker
            else:
                return None
            
    def _mem_hist(self, new_marker):
        if len(self.hist) >= self.hist_len:
            self.hist.pop(-1)
        self.hist.insert(index=0, object=new_marker)

    def get_marker_velocity(self):
        if len(self.hist) < 2:
            raise ValueError("Not enough markers detected to compute velocity")
        velos = list()
        for i in range(len(self.hist) - 1):
            velos.append(self.hist[i+1] - self.hist[i])
        return np.mean((np.array(velos), 0))

    def reset(self):
        self.marker = None
