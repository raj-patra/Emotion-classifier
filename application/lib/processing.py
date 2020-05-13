# -*- coding: utf-8 -*-
"""

@author: Raj Kishore Patra

"""

import numpy as np
import time, cv2, os, sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    print(os.path.join(base_path, relative_path))
    return os.path.join(base_path, relative_path)


class findFace(object):
    
    def __init__(self, emotions):
        
        dpath = resource_path("application/haarcascade_frontalface_default.xml")
        if not os.path.exists(dpath):
            dpath = resource_path("haarcascade_frontalface_default.xml")
        if not os.path.exists(dpath):
            print("Cascade file not present!")
            
        self.frame_in = np.zeros((10,10))
        self.frame_out = np.zeros((10,10))
        self.fps = 0
        
        self.data_buffer = []
        self.face_cascade = cv2.CascadeClassifier(dpath)
        
        self.face_rect = [1, 1, 2, 2]
        self.last_center = np.array([0, 0])
        self.last_wh = np.array([0, 0])
        self.output_dim = 13
        self.trained = False
        
        self.find_faces = True
        self.emotions = emotions
        self.last_detected = None
        
    def find_faces_toggle(self):
        self.find_faces = not self.find_faces
        return self.find_faces
    
    def shift(self, detected):
        x, y, w, h = detected
        center = np.array([x + 0.5 * w, y + 0.5 * h])
        shift = np.linalg.norm(center - self.last_center)

        self.last_center = center
        return shift
    
    def draw_rect(self, rect, col=(0, 255, 0)):
        x, y, w, h = rect
        cv2.rectangle(self.frame_out, (x, y), (x + w, y + h), col, 1)

    def get_subface_coord(self, fh_x, fh_y, fh_w, fh_h):
        x, y, w, h = self.face_rect
        return [int(x + w * fh_x - (w * fh_w / 2.0)),
                int(y + h * fh_y - (h * fh_h / 2.0)),
                int(w * fh_w),
                int(h * fh_h)]

    def get_subface_means(self, coord):
        x, y, w, h = coord
        subframe = self.frame_in[y:y + h, x:x + w, :]
        v1 = np.mean(subframe[:, :, 0])
        v2 = np.mean(subframe[:, :, 1])
        v3 = np.mean(subframe[:, :, 2])

        return (v1 + v2 + v3) / 3.
    
    def run(self):
        self.frame_out = self.frame_in
        self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in, cv2.COLOR_BGR2GRAY))
        
        if self.find_faces:
            self.data_buffer, self.trained = [], False
            detected = list(self.face_cascade.detectMultiScale(self.gray,
                                                               scaleFactor=1.3, 
                                                               minNeighbors=4, 
                                                               minSize=(100, 100), 
                                                               flags=cv2.CASCADE_SCALE_IMAGE))
            col=(100, 255, 255)
            if len(detected) > 0:
                detected.sort(key=lambda a: a[-1] * a[-2])
                if self.shift(detected[-1]) > 10:
                    self.face_rect = detected[-1]
                self.last_detected = detected
                self.emotions.predict(self.frame_in, detected)
            self.draw_rect(self.face_rect, col=(255, 0, 0))
            x, y, w, h = self.face_rect
            cv2.putText(self.frame_out, "Face",
                        (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
            return
        if set(self.face_rect) == set([1, 1, 2, 2]):
            return

        self.emotions.predict(self.frame_in, self.last_detected)
        
        