#!/usr/bin/env python
# -*- charset utf8 -*-

import numpy
import scipy as sp
import scipy.signal
from scipy import fftpack

class Params:
    def __init__(self, rate = 44100, buffer_length = 1024, mel_width = 64,
                 subdivisions = 2, m = 3, add_deltafeatures = True):
        self.rate = rate
        self.buffer_length = buffer_length
        self.mel_width = mel_width
        self.subdivisions = subdivisions
        self.m = m
        self.add_deltafeatures = add_deltafeatures

class RawSoundBuffer:
    def __init__(self):
        self.list = []
        self.offset = 0
        
    def add(self, buf):
        self.list.append(buf)
    
    def avail(self):
        return sum(c.shape[0] for c in self.list) - self.offset
    
    def read(self, buf):
        end = buf
        size = buf.shape[0]
        while len(self.list) > 0:
            cur = self.list[0][self.offset:]
            if end.shape[0] > cur.shape[0]:
                end[:cur.shape[0]] = cur
                end = end[cur.shape[0]:]
                self.offset = 0
                self.list.pop(0)
            else:
                end[:] = cur[:end.shape[0]]
                self.offset += end.shape[0]
                return
        raise Exception("ran off of end")

class Mangler:
    @staticmethod
    def f(z):
        return numpy.exp(z) - 10
    
    @staticmethod
    def inv(z):
        return numpy.log(z + 10)
    
    @staticmethod
    def diff(z):
        return numpy.exp(z)
    
    @staticmethod
    def mangle(buf, width):
        f, inv, diff = Mangler.f, Mangler.inv, Mangler.diff
        n = buf.shape[0]
        
        epsilon = 0.1
        start = inv(1 + epsilon)
        end = inv(n // 2 - epsilon)
        z = numpy.linspace(start, end, width)
        warp = f(z)
        
        _floor = numpy.floor(warp).astype(int)
        _ceil = numpy.ceil(warp).astype(int)
        _frac = warp - _floor
        
        buf = (1 - _frac) * buf[_floor] + _frac * buf[_ceil]
        buf *= diff(z)
        buf = numpy.log10(buf)
        return buf

def signal_to_spectrogram(buf, win):
    fft = fftpack.rfft(scipy.signal.detrend(buf) * win)
    pow = numpy.zeros(buf.shape[0] // 2 + 1, dtype=numpy.float32)
    fft **= 2
    pow[0] = fft[0]
    pow[1:] = fft[1::2]
    pow[1:-1] += fft[2::2]
    return pow
    
def signal_to_mel_spectrum(buf, win, width):
    return Mangler.mangle(signal_to_spectrogram(buf, win), width)
    
class generator:
    def __init__(self, params = None):
        self.params = params or Params()
        self.init()
    
    def init(self):
        buflen = self.params.buffer_length
        self.soundbuf = numpy.zeros((2*buflen), dtype=numpy.float32)
        self.raw_sound_buffer = RawSoundBuffer()
        self.raw_spectrum_buffer = []
        self.output_spectrum_buffer = []
        self.win = scipy.signal.get_window("hann", buflen, True)
        
    def add(self, buf):
        self.raw_sound_buffer.add(buf)
    
    def process(self):
        buffer_length = self.params.buffer_length
        subdivisions = self.params.subdivisions
        width = self.params.mel_width
        
        if self.raw_sound_buffer.avail() >= buffer_length:
            self.soundbuf[:buffer_length] = self.soundbuf[buffer_length:]
            self.raw_sound_buffer.read(self.soundbuf[buffer_length:])
            for j in range(1, subdivisions + 1):
                start = j * buffer_length // subdivisions
                end = start + buffer_length
                spectrum = signal_to_mel_spectrum(self.soundbuf[start:end], self.win, width)
                self.raw_spectrum_buffer.append(self.rescale(spectrum))
    
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
        if not self.params.add_deltafeatures:
            return self.next_nodeltafeatures()
        else:
            return self.next_deltafeatures()
    
    def next_nodeltafeatures(self):
        R = self.raw_spectrum_buffer
        if len(R) == 0:
            self.process()
        if len(R) == 0:
            return None
        return R.pop(0)
    
    def next_deltafeatures(self):
        m = self.params.m
        R = self.raw_spectrum_buffer
        if len(R) < 2**m + 1:
            self.process()
        if len(R) < 2**m + 1:
            return None
        a, b, c = R[0], R[m], R[2*m]
        R.pop(0)
        return numpy.concatenate((b, 2*(c-a), 2*a - 4*b + 2*c))