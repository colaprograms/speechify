#!/usr/bin/env python
# -*- charset utf8 -*-

# from https://gist.github.com/netom/8221b3588158021704d5891a4f9c0edd

import pyaudio
import numpy
import math
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy as sp
import scipy.signal

import tkinter as tk
from PIL import Image, ImageTk

RATE = 44100
BUFFER = 2048
WIDTH = 600
HEIGHT = 600
SCALE = 1

class MicrophoneDisplayer:
    def __init__(self):
        self.img = numpy.zeros((WIDTH * SCALE, HEIGHT), dtype=numpy.uint8)
        self.buf = numpy.zeros(2*BUFFER, dtype=numpy.float32)
    
    def start(self):
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=WIDTH*SCALE, height=HEIGHT)
        self.time = 0
        self.cimg = None
        self.canvas.pack()
        self.root.after(100, self.loop)
        self.startaudio()
        self.root.mainloop()
    
    def loop(self):
        self.update_line()
        self.im = Image.frombuffer('L',
            (WIDTH * SCALE, HEIGHT),
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

    def shift(self, a):
        l = len(a)
        if l == 0:
            return
        if l > 20:
            raise Exception("shifting a whole lot at once: error?")
        self.img[:, :-l] = self.img[:, l:]
        for i in range(l):
            self.img[:, -l + i] = numpy.repeat(a[i][:WIDTH], SCALE)

    def startaudio(self):
        self.py = pyaudio.PyAudio()

        self.stream = self.py.open(
            format = pyaudio.paFloat32,
            channels = 1,
            rate = RATE,
            input = True,
            output = False,
            frames_per_buffer = BUFFER
        )

    def cram(self, off):
        freq, pow = sp.signal.welch(
            self.buf[off:off+BUFFER],
            44100,
            nperseg = 2048,
            scaling = 'spectrum',
            detrend = 'linear'
        )
        return numpy.clip(
            255 / 8 * (numpy.log10(pow) + 11), 0, 255)
        
    def update_line(self):
        spec = []
        while self.stream.get_read_available() >= BUFFER:
            self.buf[:BUFFER] = self.buf[BUFFER:]
            self.buf[BUFFER:2*BUFFER] = \
                numpy.frombuffer(
                    self.stream.read(BUFFER),
                    dtype=numpy.float32)
            spec.append(self.cram(BUFFER // 2))
            spec.append(self.cram(BUFFER))
        if len(spec) > 0:
            self.shift(spec)
        #print(pow[:10])

m = MicrophoneDisplayer()
m.start()