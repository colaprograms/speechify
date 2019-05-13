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

class spectralize:
    def __init__(self, rate=44100, buffer=1024, width=64, fragments=2, m=3, add_deltafeatures=True):
        self.rate = rate
        self.buffer = buffer
        self.width = width
        self.fragments = fragments
        self.add_deltafeatures = add_deltafeatures
        self.m = m
        
        self.coeffic = None
        self.last = []
        self.buffers = []
        self.offset = 0
        self.buf = numpy.zeros((2 * self.buffer), dtype=numpy.float32)
        self.spec = []
        self.win = scipy.signal.get_window("hann", self.buffer, True)
        
    def add(self, buf):
        self.buffers.append(buf)
        
    def __avail(self):
        return sum(c.shape[0] for c in self.buffers) - self.offset
    
    def __read(self, buf):
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
    
    def process(self):
        if self.__avail() >= self.buffer:
            self.buf[:self.buffer] = self.buf[self.buffer:]
            self.__read(self.buf[self.buffer:])
            for j in range(1, self.fragments + 1):
                start = j * self.buffer // self.fragments
                end = start + self.buffer
                buf = self.buf[start:end]
                buf = self.getspectrogram(buf)
                buf = self.mangle(buf)
                buf = self.rescale(buf)
                self.last.append(buf)
    
    def getspectrogram(self, buf):
        fft = fftpack.rfft(scipy.signal.detrend(buf) * self.win)
        pow = numpy.zeros(self.buffer // 2 + 1, dtype=numpy.float32)
        fft **= 2
        pow[0] = fft[0]
        pow[1:] = fft[1::2]
        pow[1:-1] += fft[2::2]
        return pow
    
    def f(self, z):
        return numpy.exp(z) - 10
    def inv(self, z):
        return numpy.log(z + 10)
    def diff(self, z):
        return numpy.exp(z)
    
    def mangle(self, buf):
        epsilon = 0.1
        start = self.inv(1 + epsilon)
        end = self.inv(self.buffer // 2 - epsilon)
        z = numpy.linspace(start, end, self.width)
        warp = self.f(z)
        f = numpy.floor(warp).astype(int)
        c = numpy.ceil(warp).astype(int)
        b = c - warp
        buf = b * buf[f] + (1-b) * buf[c]
        buf *= self.diff(z)
        buf = numpy.log10(buf)
        return buf
    
    def rescale(self, buf):
        bottom, top = numpy.percentile(buf, [10, 100])
        #print(bottom, top)
        place = "sandbox"
        if place == "home":
            bottom, top = -4.9, 1.2
        elif place == "sandbox":
            bottom, top = -3, 4
        a, b = 255 / (top - bottom), -bottom
        return a * (buf + b)
    
    def next(self):
        if not self.add_deltafeatures:
            if len(self.last) == 0:
                self.process()
            if len(self.last) == 0:
                return None
            return numpy.clip(self.last.pop(0), 0, 255)
        else:
            if len(self.last) < 2**self.m + 1:
                self.process()
            if len(self.last) < 2**self.m + 1:
                return None
            a, b, c = self.last[0], self.last[self.m], self.last[2*self.m]
            self.last.pop(0)
            return numpy.clip(
                numpy.concatenate((b, 2*(c-a), 2*a - 4*b + 2*c)),
                0, 255)

class MicrophoneDisplayer:
    def __init__(self, width=64, height=900, add_deltafeatures=False):
        self.width = width * (3 if add_deltafeatures else 1)
        self.height = height
        self.rate = 44100
        self.add_deltafeatures = add_deltafeatures
        self.img = numpy.zeros((self.width, self.height), dtype=numpy.uint8)
        self.spe = spectralize(
            self.rate,
            width * 16,
            width,
            add_deltafeatures=add_deltafeatures
        )
        self.curline = 0
        
    def start(self):
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height)
        self.time = 0
        self.cimg = None
        self.canvas.pack()
        self.root.after(100, self.loop)
        self.startaudio()
        self.root.mainloop()
    
    def loop(self):
        self.update()
        self.im = Image.frombuffer('L',
            (self.width, self.height),
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
            cur = self.spe.next()
            if cur is None:
                break
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
        self.spe.add(numpy.frombuffer(in_data, dtype=numpy.float32))
        return (None, pyaudio.paContinue)

if __name__ == "__main__":
    m = MicrophoneDisplayer(64, 900, True)
    m.start()
