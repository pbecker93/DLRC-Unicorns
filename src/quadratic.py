from camera.camera_handler import CameraHandler
import chili_tag_detector as ctd
import cv2
import numpy as np
def measure_squareness(points):
    d1 = np.linalg.norm(points[0] - points[1])
    d2 = np.linalg.norm(points[1] - points[2])
    return d1 / d2


def preprocessor(img):
    tuple_list = ctd.detect(img)
    for t in tuple_list:
        points = t[1]
        cv2.line(img, tuple(points[0]), tuple(points[1]), [0, 0, 255])
        cv2.line(img, tuple(points[1]), tuple(points[2]), [0, 0, 255])
        print(measure_squareness(points))
    return img

ch = CameraHandler(video_capture=1, frame_rate=20, preprocessor=preprocessor)
ch.start()
