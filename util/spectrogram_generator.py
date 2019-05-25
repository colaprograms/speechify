#!/usr/bin/env python
# -*- charset utf8 -*-

import numpy
import config
import scipy.signal
from scipy import fftpack

class Params:
    def __init__(self, rate = 16000, buffer_length = 1024, mel_width = 160,
                 subdivisions = 4, m = 3, add_deltafeatures = True,
                 spectrum_range = None):
        self.rate = rate
        self.buffer_length = buffer_length
        self.mel_width = mel_width
        self.subdivisions = subdivisions
        self.m = m
        self.add_deltafeatures = add_deltafeatures
        self.spectrum_range = spectrum_range or config.microphone_volume_range

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
        return numpy.exp(z) - 60
    
    @staticmethod
    def inv(z):
        return numpy.log(z + 60)
    
    @staticmethod
    def diff(z):
        return numpy.exp(z)
    
    @staticmethod
    def mangle(buf, width):
        f, inv, diff = Mangler.f, Mangler.inv, Mangler.diff
        n = buf.shape[0]
        
        epsilon = 0.1
        start = inv(1 + epsilon)
        end = inv(n - 1 - epsilon)
        z = numpy.linspace(start, end, width)
        warp = f(z)
        
        _floor = numpy.floor(warp).astype(int)
        #_ceil = numpy.ceil(warp).astype(int)
        _frac = warp - _floor
        
        buf = (1 - _frac) * buf[_floor] + _frac * buf[_floor + 1]
        buf *= diff(z)
        buf = numpy.log10(buf + 1e-8)
        return buf
    
    @staticmethod
    def integrated_mangle(buf, width):
        f, inv, diff = Mangler.f, Mangler.inv, Mangler.diff
        n = buf.shape[0]
        
        epsilon = 0.1
        start = inv(1 + epsilon)
        end = inv(n - 1 - epsilon)
        z = numpy.linspace(start, end, width+1)
        warp = f(z)
        _floor = numpy.floor(warp).astype(int)
        _frac = warp - _floor
        def scale(z):
            return (1 - _frac) * z[_floor] + _frac * z[_floor + 1]
        def like_cumsum_but_from_the_right(z):
            return -numpy.flip(numpy.cumsum(numpy.flip(buf)))
        buf = buf.astype(numpy.float64)
        buf = like_cumsum_but_from_the_right(buf)
        buf = scale(buf)
        buf = numpy.diff(buf)
        buf = buf.astype(numpy.float32)
        buf = numpy.log10(buf + 1e-8)
        return buf

def unitlinear(m):
    dot = m * (m+1) / (m-1) / 3
    return numpy.linspace(-1, 1, m) / numpy.sqrt(dot)

def detrend(buf): # scipy's detrend is weirdly pretty slow
    buf -= numpy.mean(buf)
    x = unitlinear(buf.shape[0])
    return buf - x * numpy.dot(x, buf)

def signal_to_spectrogram(buf, win):
    fft = fftpack.rfft(buf * win)
    #fft = fftpack.rfft(detrend(buf) * win)
    pow = numpy.zeros(buf.shape[0] // 2 + 1, dtype=numpy.float32)
    fft **= 2
    pow[0] = fft[0]
    pow[1:] = fft[1::2]
    pow[1:-1] += fft[2::2]
    return pow
    
def signal_to_mel_spectrum(buf, win, width):
    return Mangler.integrated_mangle(signal_to_spectrogram(buf, win), width)
    # rar
    #return Mangler.mangle(signal_to_spectrogram(buf, win), width)

def spectrum_scale(spec, ran):
    #bottom, top = numpy.percentile(spec, [10, 100])
    bottom, top = ran#config.microphone_volume_range
    #if config.place == "home":
    #    bottom, top = -4.9, 1.2
    #elif config.place == "sandbox":
    #    bottom, top = -3, 4
    a, b = 255 / (top - bottom), -bottom
    return a * (spec + b)

def deltafeatures(spec_list, m, j=0):
    a, b, c = spec_list[j], spec_list[j+m], spec_list[j+2*m]
    return b, c-a, a - 2*b + c
    #return numpy.concatenate((b, 2*(c-a), 2*a - 4*b + 2*c))

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
        self.win = scipy.signal.get_window("blackman", buflen, True)
        #self.win = scipy.signal.get_window("hann", buflen, True)
        
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
                self.raw_spectrum_buffer.append(spectrum_scale(spectrum,
                    self.params.spectrum_range))

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
        if len(R) < 2*m + 1:
            self.process()
        if len(R) < 2*m + 1:
            return None
        ret = numpy.concatenate(deltafeatures(R, m))
        R.pop(0)
        return ret

windowcache = {}
def _window(len):
    if len not in windowcache:
        windowcache[len] = scipy.signal.get_window("hann", len, True)
    return windowcache[len]

class whole_buffer:
    def __init__(self, params = None):
        self.params = params or Params()
        self.buflen = self.params.buffer_length
        self.win = _window(self.params.buffer_length)
        self.run = False
        
    def set(self, buf):
        self.buf = buf
        self.offset = 0
        self.raw_spectrum_buffer = []
        self.run = True
    
    def process(self):
        if self.offset + self.buflen > self.buf.shape[0]:
            self.run = False
        if self.run:
            spectrum = signal_to_mel_spectrum(
                self.buf[self.offset:self.offset + self.buflen],
                self.win,
                self.params.mel_width)
            self.raw_spectrum_buffer.append(spectrum_scale(spectrum,
                self.params.spectrum_range))
            self.offset += self.buflen // 2
    
    def next(self):
        R = self.raw_spectrum_buffer
        while self.run and len(R) < 2*self.params.m + 1:
            self.process()
        if self.run:
            ret = numpy.stack(deltafeatures(R, self.params.m), axis=1)
            R.pop(0)
            return ret
        return
    
    def all(self, buf):
        self.set(buf)
        out = []
        while self.run:
            z = self.next()
            if z is not None:
                out.append(z)
        if len(out) > 0:
            return numpy.stack(out)
        else:
            raise Exception("problem: " + str(buf.shape))
