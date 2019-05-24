import tensorflow as tf
import numpy

from util.data import LibriSpeech
from util.spectrogram_generator import whole_buffer

from nets.model import EncoderDecoder
import soundfile as sf
from os.path import join

import util.onehot as onehot
import config

class bufmixer:
    z = numpy.linspace(0, 1, 160)
    bufmixer0 = numpy.maximum(2*z - 1, 0)[:, None]
    bufmixer1 = 1 - numpy.abs(2*z - 1)[:, None]
    bufmixer2 = numpy.maximum(1 - 2*z, 0)[:, None]
    
    @staticmethod
    def mixer(buf):
        return numpy.concatenate([buf * bufmixer.bufmixer0,
                                  buf * bufmixer.bufmixer1,
                                  buf * bufmixer.bufmixer2], axis=2)

ENDPAD = 6
BUFPAD = 4

def _create(get, start, end):
    bufs = []
    transs = []
    longest_buf = 0
    longest_trans = 0
    for i in range(0, end - start):
        trans, buf = get(start + i)
        assert buf.shape[1:] == (160, 3)
        longest_buf = max(longest_buf, buf.shape[0])
        longest_trans = max(longest_trans, len(trans))
        bufs.append(buf)
        transs.append(trans)
    longest_buf = ((longest_buf + BUFPAD - 1) // BUFPAD) * BUFPAD
    bufmatrix = numpy.zeros((len(bufs), longest_buf, 160, 9))
    for i, buf in enumerate(bufs):
        bufmatrix[i, :buf.shape[0], :, :] = bufmixer.mixer(buf)
    longest_trans += 1 + ENDPAD
    transmatrix = numpy.zeros((len(transs), longest_trans, onehot.nchars))
    transoffset = numpy.zeros_like(transmatrix)
    for i, trans in enumerate(transs):
        trans = "@" + trans + "$" * (longest_trans - len(trans) - 1)
        for j, c in enumerate(trans):
            transmatrix[i, j, onehot.idx[c]] = 1
    transoffset[:, :-1, :] = transmatrix[:, 1:, :]
    transoffset[:, -1, onehot.idx["$"]] = 1
    return (bufmatrix, transmatrix), transoffset

class SequenceFromLibriSpeech(tf.keras.utils.Sequence):
    def __init__(self, dat, batchsize, get):
        self.data = dat
        self.batchsize = batchsize
        #self.wb = wholebuffer
        self.get = get
    
    def __len__(self):
        return (len(self.data) + self.batchsize - 1) // self.batchsize

    def __getitem__(self, idx):
        start = idx * self.batchsize
        end = min((idx + 1) * self.batchsize, len(self.data))
        print("returning batch %d %d" % (start, end))
        retval = _create(self.get, start, end)
        return retval
        
class LibriSequence:
    def __init__(self):
        self.path = config.path
        self.ls = LibriSpeech()
        self.ls.load()
        self.wb = whole_buffer()
        self.wb.params.spectrum_range = config.librispeech_range
        self.batchsize = 32
    
    def sequence(self, type="train"):
        def get(ix):
            reader,book,i = self.ls.info[type][ix]
            file = self.ls.data[reader][book][i]
            buf, _ = sf.read(join(self.path, file['path']))
            trans = file['trans']
            print("buffer shape", buf.shape)
            return trans, self.wb.all(buf)
        return SequenceFromLibriSpeech(self.ls.info['train'], self.batchsize, get)

"""
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
                transs.append(trans)
            bufmatrix = numpy.zeros((len(bufs), longest_buf, 160, 9))
            for i, buf in enumerate(bufs):
                bufmatrix[i, :buf.shape[0], :, :3] = buf * bufmixer0[:, None]
                bufmatrix[i, :buf.shape[0], :, 3:6] = buf * bufmixer1[:, None]
                bufmatrix[i, :buf.shape[0], :, 6:9] = buf * bufmixer2[:, None]
            longest_trans += 1 + FancySampler.ENDPAD
            transmatrix = numpy.zeros((len(transs), longest_trans, onehot.nchars))
            for i, trans in enumerate(transs):
                trans = "@" + trans + "$" * (longest_trans - len(trans) - 1)
                for j, c in enumerate(trans):
                    transmatrix[i, j, onehot.idx[c]] = 1
            print("yielding")
            yield bufmatrix, transmatrix
"""

def train():
    encdec = EncoderDecoder()
    spectrum = tf.keras.layers.Input((None, 160, 9))
    transcript = tf.keras.layers.Input((None, len(onehot.chars)))
    decode = encdec(spectrum, transcript)
    #decode = tf.nn.softmax(decode)
    model = tf.keras.models.Model([spectrum, transcript], decode)
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0, nesterov=True),
                  loss = 'categorical_crossentropy')
    samp = LibriSequence()
    model.fit_generator(
        samp.sequence("train"),
        steps_per_epoch = 1000,
        epochs = 10,
        verbose = 2,
        validation_data = samp.sequence("test"),
        validation_steps = 10,
        workers = 1
    )