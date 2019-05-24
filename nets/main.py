import tensorflow as tf
import numpy

from util.sample import Sampler
from util.data import LibriSpeech
from util.spectrogram_generator import whole_buffer

from nets.model import EncoderDecoder
import soundfile as sf
from os.path import join

import util.onehot as onehot

class FancySampler:
    ENDPAD = 6
    
    def __init__(self):
        self.path = config.path
        self.ls = LibriSpeech()
        self.ls.load()
        self.wb = whole_buffer()
        self.wb.params.spectrum_range = config.librispeech_range
        self.batchsize = 32
    
    def get(self, type="train"):
        if type == "train":
            file = self.ls.uniform_train()
        else:
            file = self.ls.uniform_test()
        buf, _ = sf.read(join(self.path, file['path']))
        trans = file['trans']
        return trans, self.wb.all(buf)
    
    def generate(self, type="train"):
        z = numpy.linspace(0, 1, 160)
        bufmixer0 = numpy.maximum(2*z - 1, 0)
        bufmixer1 = 1 - numpy.abs(2*z - 1)
        bufmixer2 = numpy.maximum(1 - 2*z, 0)
        while True:
            bufs = []
            transs = []
            longest_buf = 0
            longest_trans = 0
            for i in range(self.batchsize):
                trans, buf = self.get(type)
                assert buf.shape[1:] == (160, 3)
                longest_buf = max(longest_buf, buf.shape[0])
                longest_trans = max(longest_trans, len(trans))
                bufs.append(buf)
                transcript_buffer = numpy.zeros((len(trans), onehot.nchars))
                for i, c in enumerate(trans):
                    transcript_buffer[i, onehot.idx[c]] = 1
                transs.append(transcript_buffer)
            bufmatrix = numpy.zeros((len(bufs), longest_buf, 160, 9))
            for i, buf in enumerate(bufs):
                bufmatrix[i, :buf.shape[0], :, :3] = buf * bufmixer0
                bufmatrix[i, :buf.shape[0], :, 3:6] = buf * bufmixer1
                bufmatrix[i, :buf.shape[0], :, 6:9] = buf * bufmixer2
            transmatrix = numpy.zeros((len(transs), longest_trans, onehot.nchars))
            for i, trans in enumerate(transs):
                trans = "@" + trans + "$" * (longest_trans - len(trans) + FancySampler.ENDPAD)
                for j, c in enumerate(trans):
                    transmatrix[i, j, onehot.idx[c]] = 1
        return bufmatrix, transmatrix

def train():
    encdec = EncoderDecoder()
    spectrum = tf.keras.layers.Input((None, None, 160, 9))
    transcript = tf.keras.layers.Input((None, None, len(onehot.chars)))
    decode = encdec(transcript, spectrum)
    model = tf.keras.models.Model([spectrum, transcript], [decode])
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0, nesterov=True),
                  loss = 'categorical_crossentropy')
    samp = FancySampler()
    model.fit_generator(
        samp.generate("train"),
        steps_per_epoch = 1000,
        epochs = 10,
        verbose = 2,
        validation_data = samp.generate("test"),
        validation_freq = 1,
        workers = 4,
        use_multiprocessing = True
    )