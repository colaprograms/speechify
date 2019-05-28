from util.spectrogram_generator import whole_buffer
from util.data import LibriSpeech
import soundfile as sf
from os.path import join
import matplotlib.pyplot as p

ls = LibriSpeech()
ls.load()
file = ls.uniform_random()
path = join(r"c:\users\meta\documents\speechify\LibriSpeech\combined", file['path'])
print("path:", file['path'])
print("trans:", file['trans'])

wb = whole_buffer()
buf, _ = sf.read(path)
out = wb.all(buf)
p.imshow(out)
p.show()