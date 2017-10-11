from camera import CameraHandler
#from camera.camera_handler_2 import CameraHandler
from policy import Detector
import numpy as np
from fcn import FCN
import cv2
'''FCN'''
VGG_WEIGHTS = '/home/nvidia/RobotControl/src/fcn/vgg16.npy'
MODEL_PATH = '/home/nvidia/RobotControl/src/fcn/model_save/old_2/'

model = FCN((240, 320), 1, VGG_WEIGHTS)
model.load(path=MODEL_PATH)


img = None
ct = 0

mask=np.zeros((480, 640), dtype=np.uint8)
mask = cv2.fillConvexPoly(mask, np.array([[275, 285], [365, 285], [430, 166], [210, 166]]), 1, 1)


def preprocessor(img):
#    mask=np.zeros((480, 640), dtype=np.uint8)
#    mask = cv2.fillConvexPoly(mask, np.array([[275, 285], [365, 285], [430, 166], [210, 166]]), 1, 1)
    return model.extract_hmap(img)[0]*mask
  #  return img * np.expand_dims(mask,-1)
handler = CameraHandler(2, frame_rate=20, preprocessor=lambda img: preprocessor(img))
handler.start()
# img = handler.get_next_frame()
# cv2.imwrite("crop1.jpg",img)
