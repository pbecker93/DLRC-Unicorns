import os
import cv2
import numpy as np
from threading import Thread, Lock
import time

class CameraHandler(Thread):

    def __init__(self, video_capture=1, frame_rate=40, preprocessor=None):
        super().__init__()
        self.video_capture = video_capture
        self.frame_rate = frame_rate
        self.cap = cv2.VideoCapture(video_capture)
        self.cap.set(cv2.CAP_PROP_FPS, frame_rate) 
        print("CALI")
        self._calibrate_lighting()
        self.paused = False
        self.recording = False
        self.recording_lock = Lock()
        self.frame_read_lock = Lock()
        self.frame_read_lock.acquire()
        self.preporcessor = preprocessor if preprocessor is not None else (lambda img: img)
        self.darkness=0

        self.fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        #self.out = cv2.VideoWriter('output.avi', self.fourcc, 20.0, (640, 480))

    def run(self):

        while (self.cap.isOpened()):
            ret, raw_frame = self.cap.read()

            self.frame = self.preporcessor(raw_frame)
            if self.frame_read_lock.locked():
                self.frame_read_lock.release()

            if ret == True:
                if self.recording:
                    self.recording_lock.acquire()
                    self.out.write(self.frame)
                    self.recording_lock.release()
                if self.video_capture != 1:
                    cv2.imshow('Camera: ' + str(self.video_capture), self.frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                break

    def start_recording(self, filename):
        print("Camera Handler: Start Recording")
        self.out = cv2.VideoWriter(filename + ".avi",
                                   self.fourcc,
                                   self.frame_rate,
                                   (640, 480))
        self.recording = True

    def _get_calibration_frame(self, img):
        if self.video_capture == 1:
            return img[320:460,160:480]
        if self.video_capture == 2:
            return img[120:260,160:480]

    def _calibrate_lighting(self):
        self.exp=180
        self.gain=0
        os.system("v4l2-ctl -d /dev/video"+str(self.video_capture)+" -c gain_automatic=0")
        os.system("v4l2-ctl -d /dev/video"+str(self.video_capture)+" -c gain="+str(self.gain) )
        os.system("v4l2-ctl -d /dev/video"+str(self.video_capture)+" -c auto_exposure=1")
        os.system("v4l2-ctl -d /dev/video"+str(self.video_capture)+" -c exposure="+str(self.exp) )
        darkness = 0
        frame_num = 0
        if self.video_capture == 1:
            upper_thresh = 100
            lower_thresh = 90
        elif self.video_capture ==  2:
            upper_thresh = 80
            lower_thresh = 70

        while (self.cap.isOpened()) and frame_num<50:

            ret, raw_frame = self.cap.read()
            gauge = self._get_calibration_frame(raw_frame)
            cv2.imshow("Gauge",gauge)
            darkness = np.mean(cv2.cvtColor(gauge, cv2.COLOR_BGR2HLS)[:,:,1])
            frame_num+=1
            print(darkness)
                 
            if darkness>upper_thresh:
                if self.gain !=0:
                   self.gain = self.gain-1
                   os.system("v4l2-ctl -d /dev/video"+str(self.video_capture)+" -c gain="+str(self.gain) ) 
                else: 
                    self.exp-=5
                    os.system("v4l2-ctl -d /dev/video"+str(self.video_capture)+" -c exposure="+str(self.exp) )
            elif darkness<lower_thresh:
                if self.exp>250:
                    self.gain = self.gain+1
                    os.system("v4l2-ctl -d /dev/video"+str(self.video_capture)+" -c gain="+str(self.gain) ) 
                else:
                     self.exp+=5
                     os.system("v4l2-ctl -d /dev/video"+str(self.video_capture)+" -c exposure="+str(self.exp) )

            if self.gain==15:
                break
            if lower_thresh<darkness<upper_thresh:
                break

    def stop_recording(self):
        self.recording_lock.acquire()
        print("Camera Handler: Stop Recording")
        self.recording = False
        self.out.release()
        self.recording_lock.release()

    @property
    def current_img(self):
        return self.frame

    def get_next_frame(self):
        self.frame_read_lock.acquire()
        return self.frame

if __name__ == '__main__':

    cam_handler = CameraHandler(2)
    cam_handler.start()
    cv2.waitKey(1)
