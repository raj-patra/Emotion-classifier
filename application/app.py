# -*- coding: utf-8 -*-
"""

@author: Raj Kishore Patra

"""
__version__ = "1.0.1"

from lib.cam import Camera
import cv2
from tkinter import *

from lib.processing import findFace
from lib.emotions import Emotions
from PIL import Image, ImageTk

from text_strings import *


class Pulse(object):
    
    def __init__(self):
        self.camera = Camera(camera=0)
        if not self.camera.valid:
            raise ValueError('Camera Not Working!')
        
        self.w, self.h, self.e = 0, 0, Emotions()
        self.processor = findFace(emotions = self.e)
        
        self.logged_in = False
        self.sending_emotions = False
        
    def start(self):
        self.processor.find_faces_toggle()
        
    def loop(self):
        frame = self.camera.get_frame()
        frame = cv2.flip(frame, 1)
        
        self.h, self.w, _c = frame.shape
        self.processor.frame_in = frame
        self.processor.run()
        
        output_frame = self.processor.frame_out
        output_frame = cv2.resize(output_frame, (800, 550))
        cv2image=cv2.cvtColor(output_frame,cv2.COLOR_BGR2RGBA)
        
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        
        set_emotions_labels(self.e.get_last_prediction())
        
def set_emotions_labels(emotions):
	for i in range(7):
		labels_emotion[i].configure(font=('Papyrus', 12, 'bold'), fg='black')
		labels_emotions_value[i].configure(font=('Papyrus', 12, 'bold'), fg='black')
		e = emotions[i]
		e = "{:0.2f}%".format(e * 100)
		var_emotions[i].set(e)

	max_emotion = emotions.argmax()
	labels_emotion[max_emotion].configure(font=('Papyrus', 12, 'bold'), fg='red')
	labels_emotions_value[max_emotion].configure(font=('Papyrus', 12, 'bold'), fg='red')


p = Pulse()
root = Tk()
root.title('Emotion Detector')
root.resizable(0,0)

# ----------------- FRAME EMOTIONS ------------------ #

def insert_row(frame=None, index=None, text=None, text_var=None):
	l1 = Label(frame, borderwidth=2, relief="groove", width=22, height=2, text=text, bg='white')
	l1.grid(column=0, row=index+3, sticky=E + W + N + S)
	l2 = Label(frame, borderwidth=2, relief="groove", width=10, height=2, textvariable=text_var, bg='white')
	l2.grid(column=1, row=index+3, sticky=E + W + N + S)
	return l1, l2


frame_emotions = Frame(root, bd=5, bg='white')
frame_emotions.grid(column=0, row=0, rowspan=50,  pady=5, padx=5)
frame_emotions.lower()

labels_emotion = [None] * 7
labels_emotions_value = [None] * 7
var_emotions = [StringVar() for i in range(7)]

emotions_l = ['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Suprise', 'Indifference']


# row for each emotion
for i in range(7):
	var_emotions[i].set("")
	(labels_emotion[i], labels_emotions_value[i]) = insert_row(frame=frame_emotions,
    index=i, text=emotions_l[i],text_var=var_emotions[i])
															 
# ----------------- FRAME EMOTIONS END -------------- #
        
# ----------------- FRAME MAIN VIDEO ---------------- #

# main video stream
lmain = Label(root, height=400, bd=5)
lmain.grid(column=1, row=0, columnspan=4, rowspan=17, sticky=E + W + N + S)
lmain.lower()

# ----------------- FRAME MAIN VIDEO END ------------ #


while True:
	p.loop()
	root.update_idletasks()
	root.update()
       
        
        
        
        
        
        
        
        
        
        
        
        