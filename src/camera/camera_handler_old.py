import os
import cv2
from threading import Thread, Lock
import time

class CameraHandler(Thread):

    def __init__(self, video_capture=1, frame_rate=40, preprocessor=None):
        super().__init__()
        self.video_capture = video_capture
        self.frame_rate = frame_rate
        self.cap = cv2.VideoCapture(video_capture)
        self.cap.set(cv2.CAP_PROP_FPS, frame_rate)
        os.system("v4l2-ctl -d /dev/video"+str(video_capture)+" -c gain_automatic=0")
        os.system("v4l2-ctl -d /dev/video"+str(video_capture)+" -c gain=10")
        os.system("v4l2-ctl -d /dev/video"+str(video_capture)+" -c auto_exposure=1")
        os.system("v4l2-ctl -d /dev/video"+str(video_capture)+" -c exposure=230")

        self.paused = False
        self.recording = False
        self.recording_lock = Lock()
        self.frame_read_lock = Lock()
        self.frame_read_lock.acquire()
        self.preporcessor = preprocessor if preprocessor is not None else (lambda img: img)


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
                if self.video_capture != 2:
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

    cam_handler = CameraHandler()
    cam_handler.start()
    time.sleep(3)
    cam_handler.start_recording("output")
    time.sleep(4)
    cv2.imwrite('dummy.png', cam_handler.current_img)
    time.sleep(4)
    cam_handler.stop_recording()
