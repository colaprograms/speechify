import pyaudio
import wave
import sys
import numpy as np

p = pyaudio.PyAudio()
s = p.open(44100, 2, pyaudio.paInt8, input=True)
while True:
    x = s.read(4096)
    z = np.fromstring(x, dtype=np.int8)
    print(z.sum(), s.get_read_available())