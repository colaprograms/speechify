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
HEIGHT = 900
SCALE = 2

class MicrophoneDisplayer:
    def __init__(self):
        self.img = numpy.zeros((WIDTH * SCALE, HEIGHT), dtype=numpy.uint8)
        self.buf = numpy.zeros(2*BUFFER, dtype=numpy.float32)
        self.buffers = []
        self.offset = 0
    
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
        self.root.after(100, self.loop)

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
            frames_per_buffer = BUFFER,
            stream_callback = self.callback
        )

        self.stream.start_stream()

    def callback(self, in_data, frame_count, time_info, status_flags):
        self.buffers.append(
            numpy.frombuffer(in_data, dtype=numpy.float32))
        return (None, pyaudio.paContinue)

    def get_read_available(self):
        return sum(c.shape[0] for c in self.buffers) - self.offset
    
    def read(self, buf):
        off = 0
        size = buf.shape[0]
        while len(self.buffers) > 0:
            end = buf[off:]
            cur = self.buffers[0][self.offset:]
            if end.shape[0] > cur.shape[0]:
                end[:cur.shape[0]] = cur
                self.offset = 0
                self.buffers.pop(0)
            else:
                end[:] = cur[:end.shape[0]]
                self.offset += end.shape[0]
                return
        raise Exception("ran off of end")

    def cram(self, off):
        freq, pow = sp.signal.welch(
            self.buf[off:off+BUFFER],
            44100,
            nperseg = 2048,
            scaling = 'spectrum',
            detrend = 'linear'
        )
        #print(numpy.log10(pow[:12]))
        return numpy.clip(
            255 / 7 * (numpy.log10(pow) + 11), 0, 255)
        
    def update_line(self):
        spec = []
        #print(self.get_read_available())
        #print(self.buffers)
        while self.get_read_available() >= BUFFER:
            self.buf[:BUFFER] = self.buf[BUFFER:]
            self.read(self.buf[BUFFER:2*BUFFER])
            spec.append(self.cram(BUFFER // 2))
            spec.append(self.cram(BUFFER))
        if len(spec) > 0:
            self.shift(spec)
        #print(pow[:10])

m = MicrophoneDisplayer()
m.start()
