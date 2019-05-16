from util.spectrogram_generator import whole_buffer
from util.data import LibriSpeech
import soundfile as sf
from os.path import join

class Sampler:
    def __init__(self, home=False):
        if home:
            self.path = "../speechify_dat/combined"
        else:
            self.path = r"c:\users\meta\documents\speechify\LibriSpeech\combined"
        self.ls = LibriSpeech()
        self.ls.load()
        self.wb = whole_buffer()
    
    def rand(self):
        file = self.ls.uniform_random()
        path = join(self.path, file['path'])
        buf, _ = sf.read(path)
        return file, self.wb.all(buf)
    
    def whatev(self):
        pass

def imshow(buf):
    import matplotlib.pyplot as p
    p.imshow(buf)
    p.show()