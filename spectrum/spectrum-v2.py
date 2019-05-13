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

from scipy import fftpack

import tkinter as tk
from PIL import Image, ImageTk

RATE = 44100
if True:
    BUFFER = 1024
    WIDTH = 64#1025
    MEL_WIDTH = 64
else:
    BUFFER = 2048
    WIDTH = 128
    MEL_WIDTH = 128
FRAGMENTS = 2
add_deltafeatures = True
if add_deltafeatures:
    WIDTH *= 3
HEIGHT = 900
SCALE = 1
M = 3
MODE = "scan"

class MicrophoneDisplayer:
    def __init__(self):
        self.img = numpy.zeros((WIDTH * SCALE, HEIGHT), dtype=numpy.uint8)
        self.buf = numpy.zeros(2*BUFFER, dtype=numpy.float32)
        self.buffers = []
        self.offset = 0
        self.curline = 0
        self.win = scipy.signal.get_window("hann", BUFFER, True)
        self.coeffic = None
        self.last = []
    
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

    def row(self, a):
        if not add_deltafeatures:
            return numpy.clip(a, 0, 255)
        self.last.append(a)
        if len(self.last) < 2**M + 1:
            return numpy.zeros(3 * MEL_WIDTH)
        a, b, c = self.last[0], self.last[M], self.last[2*M]
        self.last.pop(0)
        return numpy.clip(
            numpy.concatenate((b, 2*(c-a), 2*a - 4*b + 2*c)),
            0, 255)
        
    def shift(self, a):
        l = len(a)
        if l == 0:
            return
        #if l > 90:
        #    print("shifting a lot:", l)
            #if l > 180:
            #    raise Exception("shifting a whole lot at once: error?")
        if MODE == "roll":
            self.img[:, :-l] = self.img[:, l:]
            for i in range(l):
                self.img[:, -l + i] = self.row(a[i][:WIDTH])
        elif MODE == "scan":
            for i in range(l):
                self.img[:, HEIGHT - 1 - self.curline] = self.row(a[i][:WIDTH])
                self.curline += 1
                if self.curline == HEIGHT:
                    self.curline = 0
            self.img[:, HEIGHT - 1 - self.curline] = 0
        else:
            raise Exception("unknown mode")
    
    def startaudio(self):
        self.py = pyaudio.PyAudio()

        self.stream = self.py.open(
            format = pyaudio.paFloat32,
            channels = 1,
            rate = RATE,
            input = True,
            output = False,
            frames_per_buffer = 1024,
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
        end = buf
        size = buf.shape[0]
        while len(self.buffers) > 0:
            cur = self.buffers[0][self.offset:]
            if end.shape[0] > cur.shape[0]:
                end[:cur.shape[0]] = cur
                end = end[cur.shape[0]:]
                self.offset = 0
                self.buffers.pop(0)
            else:
                end[:] = cur[:end.shape[0]]
                self.offset += end.shape[0]
                return
        raise Exception("ran off of end")

    def cram(self, off):
        buf = scipy.signal.detrend(self.buf[off:off+BUFFER])
        fft = fftpack.rfft(buf * self.win)
        pow = numpy.zeros(BUFFER // 2 + 1, dtype=numpy.float32)
        pow[0] = fft[0] ** 2
        pow[1:] = fft[1::2] ** 2
        pow[1:-1] += fft[2::2] ** 2
        return self.mangle(pow)
        #return numpy.clip(
        #    255 / 7 * (numpy.log10(pow) + 4), 0, 255)

    def f(self, z):
        return numpy.exp(z) - 10
    def inv(self, z):
        return numpy.log(z + 10)

    def mangle(self, pow):
        #warp = numpy.exp(numpy.linspace(1, numpy.log(1024 - 0.1), 128))
        warp = self.f(numpy.linspace(
            self.inv(1 + 0.1),
            self.inv(BUFFER // 2 - 0.1),
            MEL_WIDTH))
        f = numpy.floor(warp).astype(int)
        c = numpy.ceil(warp).astype(int)
        b = c - warp
        pow = b * pow[f] + (1-b) * pow[c]
        pow *= warp
        pow = numpy.log10(pow)
        bottom, top = numpy.percentile(pow, [10, 100])
        place = "sandbox"
        if place == "home":
            bottom = -4.9
            top = 1.2
        elif place == "sandbox":
            bottom = -3
            top = 4
        self.coeffic = 255/(top - bottom), -bottom
        a, b = self.coeffic
        return a * (pow + b)#return numpy.clip(a * (pow + b), 0, 255)
        
    def update_line(self):
        spec = []
        #print(self.buffers)
        while self.get_read_available() >= BUFFER:
            self.buf[:BUFFER] = self.buf[BUFFER:]
            self.read(self.buf[BUFFER:])
            #spec.append(self.cram(BUFFER // 2))
            #spec.append(self.cram(BUFFER))#for j in range(1, 9):
            for j in range(1, FRAGMENTS + 1):
                spec.append(self.cram(j * BUFFER // FRAGMENTS))
            #for j in range(1, 9):
            #    spec.append(self.cram(j * BUFFER // 8))#    spec.append(self.cram(j * BUFFER // 8))
        if len(spec) > 0:
            self.shift(spec)
        #print(pow[:10])

m = MicrophoneDisplayer()
m.start()
