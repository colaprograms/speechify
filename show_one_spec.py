from util.spectrogram_generator import whole_buffer
from util.data import LibriSpeech
import soundfile as sf
from os.path import join
import matplotlib.pyplot as p
import numpy as np
import config

ls = LibriSpeech()
ls.load()
file = ls.uniform_train()
path = join(r"c:\users\meta\documents\speechify\LibriSpeech\combined", file['path'])
print("path:", file['path'])
print("trans:", file['trans'])

wb = whole_buffer()
wb.params.spectrum_range = config.librispeech_range
buf, _ = sf.read(path)
out = wb.all(buf)
con = np.concatenate([out[:, :, 0], out[:, :, 1], out[:, :, 2]], axis=1)
m, x = np.min(out[:, :, 0]), np.max(out[:, :, 0])
con = (con - m) / (x - m)
p.imshow(con)
p.show()