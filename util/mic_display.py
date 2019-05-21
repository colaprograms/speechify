#!/usr/bin/env python
# -*- charset utf8 -*-

# from https://gist.github.com/netom/8221b3588158021704d5891a4f9c0edd

import pyaudio
import numpy
import tkinter as tk
from PIL import Image, ImageTk

from util.spectrogram_generator import Params, generator

VERBOSE = True
class MicrophoneDisplayer:
    def __init__(self, rate=16000, width=64, add_deltafeatures=False):
        height = 900
        self.width = width
        self.imgwidth = width * (3 if add_deltafeatures else 1)
        self.height = height
        self.rate = rate
        self.add_deltafeatures = add_deltafeatures
        self.img = numpy.zeros((self.imgwidth, self.height), dtype=numpy.uint8)
        # we are aiming for 15~20 ms per buffer
        if self.rate == 16000:
            self.fftwidth = 512 # 16 ms
        elif self.rate == 44100:
            self.fftwidth = 1024 # 11 ms
        else:
            raise Exception("don't know the fftwidth for this rate")
        self.params = Params(self.rate, self.fftwidth, width, add_deltafeatures = add_deltafeatures)
        self.params.subdivisions = 4
        self.generator = generator(self.params)
        self.curline = 0
        if VERBOSE:
            print("Created microphone display.")
            print("Signal rate: %d Hz" % self.rate)
            print("FFT width: %d" % self.fftwidth)
            print("Time between buffers: %d ms" % (self.time_between_buffers() * 1000))
    
    def time_between_buffers(self):
        # samples per buffer * seconds per buffer / subdivisions
        return self.fftwidth / self.rate / self.params.subdivisions
    
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
