"Convert all of the flacs to spectrograms."

import os, config
from util.data import LibriSpeech, FilesToSpectrogram
from os.path import join

def mkdir(*path):
    path = join(*path)
    if not os.path.exists(path):
        os.mkdir(path)
        print("Making directory %s" % path)

def convert_all_flac():
    path = config.path
    dest = "../speechify_dat/spectrogram"
    ls = LibriSpeech()
    ls.load()
    f = FilesToSpectrogram()
    for reader, r_ in ls.data.items():
        mkdir(dest, reader)
        for book, b_ in r_.items():
            mkdir(dest, reader, book)
            for i, file in enumerate(b_):
                npzf = join(dest, reader, book, "%s-%s-%04d.npy" % (reader, book, i))
                print("Creating %s" % npzf)
                f.write_out_spectrogram(npzf, join(path, file['path']))

if __name__ == "__main__":
    convert_all_flac()
