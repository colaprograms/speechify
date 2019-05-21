from util.spectrogram_generator import whole_buffer
from util.data import LibriSpeech
import soundfile as sf
from os.path import join
import matplotlib.pyplot as p
import config
import numpy

class Sampler:
    def __init__(self, home=False):
        self.path = config.path
        self.ls = LibriSpeech()
        self.ls.load()
        self.wb = whole_buffer()
        self.wb.params.spectrum_range = config.librispeech_range
    
    def rand(self):
        file = self.ls.uniform_train() #file = self.ls.uniform_random()
        buf, _ = sf.read(join(self.path, file['path']))
        return file, self.wb.all(buf)
    
    def whatev(self):
        pass

def time(m):
    sa = Sampler()
    import time
    start = time.time()
    for i in range(m):
        sa.rand()
    return time.time() - start

def imshow(buf):
    # rar
    # rar
    import matplotlib
    import matplotlib.pyplot as p
    p.figure(figsize=(20,100))
    flat = numpy.concatenate([buf[:, :, i] for i in range(buf.shape[2])], axis=1)
    p.imshow(flat, norm=matplotlib.colors.Normalize(0, 255))
    p.show()
