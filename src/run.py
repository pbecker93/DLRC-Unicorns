from policy.lego_policy import Policy
from mqtt_master.master import Master
from camera.camera_handler import CameraHandler
from fcn import FCN


'''MQTT MASTER'''

#DLRC
#BROKER_IP = "192.168.179.40"

BROKER_IP = "10.250.144.181"
BROKER_PORT = 1883

'''CAMERA INPUTS'''
CAM_FRONT = 1
CAM_DOWN = 2

'''FCN'''
VGG_WEIGHTS = '/home/nvidia/RobotControl/src/fcn/vgg16.npy'
MODEL_PATH = '/home/nvidia/RobotControl/src/fcn/model_save/old_2/'



master = Master(BROKER_IP, BROKER_PORT)

cam_handler_front = CameraHandler(video_capture=CAM_FRONT,
                                  frame_rate=20)
cam_handler_down = CameraHandler(video_capture=CAM_DOWN,
                                 frame_rate=20)

model = FCN((240, 320), 1, VGG_WEIGHTS)
model.load(path=MODEL_PATH)

cam_handler_front.start()
cam_handler_down.start()
pol = Policy(master, model, cam_handler_front, cam_handler_down)
