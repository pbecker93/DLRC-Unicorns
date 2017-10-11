import cv2
from camera.camera_handler import CameraHandler
from policy.lego_policy import Policy
def preprocessor(img):
    img = cv2.line(img, (320, 0), (320, 480), [255, 0, 0])
    img = cv2.line(img, (int(Policy.THRES_L(0)), 0), (int(Policy.THRES_L(480)), 480), [0, 255, 0])
    img = cv2.line(img, (int(Policy.THRES_R(0)), 0), (int(Policy.THRES_R(480)), 480), [0, 0, 255])
    return img

base_line = 30
factor = 0.5


handler = CameraHandler(1, frame_rate=20, preprocessor=preprocessor)
handler.start()
