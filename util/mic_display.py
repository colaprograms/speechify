#!/usr/bin/env python
# -*- charset utf8 -*-

# from https://gist.github.com/netom/8221b3588158021704d5891a4f9c0edd

import pyaudio
import numpy
import tkinter as tk
from PIL import Image, ImageTk

from util.spectrogram_generator import Params, generator

class MicrophoneDisplayer:
    def __init__(self, width=64, height=900, add_deltafeatures=False):
        self.width = width
        self.imgwidth = width * (3 if add_deltafeatures else 1)
        self.height = height
        self.rate = 44100
        self.add_deltafeatures = add_deltafeatures
        self.img = numpy.zeros((self.imgwidth, self.height), dtype=numpy.uint8)
        self.generator = generator(
            Params(self.rate, width * 16, width, add_deltafeatures = add_deltafeatures)
        )
        self.curline = 0
        
    def start(self):
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=self.imgwidth, height=self.height)
        self.time = 0
        self.cimg = None
        self.canvas.pack()
        self.root.after(100, self.loop)
        self.startaudio()
        self.root.mainloop()
    
    def loop(self):
        self.update()
        self.im = Image.frombuffer('L',
            (self.imgwidth, self.height),
            self.img.T.tobytes(),
            "raw"
        )
        self.photo = ImageTk.PhotoImage(image = self.im)
        
        if self.cimg is None:
            self.cimg = self.canvas.create_image(
                0,
                0,
                image = self.photo,
                anchor = tk.NW)
        else:
            #print("cimg", self.cimg)
            self.canvas.itemconfig(
                self.cimg,
                image = self.photo
            )
        #print("loop")
        self.root.after(10, self.loop)

    def update(self):
        while True:
            cur = self.generator.next()
            if cur is None:
                break
            cur = numpy.clip(cur, 0, 255)
            self.img[:, self.height - 1 - self.curline] = cur
            self.curline += 1
            if self.curline == self.height:
                self.curline = 0
        self.img[:, self.height - 1 - self.curline] = 0
            
    def startaudio(self):
        self.py = pyaudio.PyAudio()

        self.stream = self.py.open(
            format = pyaudio.paFloat32,
            channels = 1,
            rate = self.rate,
            input = True,
            output = False,
            frames_per_buffer = 1024,
            stream_callback = self.callback
        )

        self.stream.start_stream()

    def callback(self, in_data, frame_count, time_info, status_flags):
        # rar
        self.generator.add(numpy.frombuffer(in_data, dtype=numpy.float32))
        return (None, pyaudio.paContinue)