import tensorflow as tf
import numpy
from util.data import LibriSpeech
from util.spectrogram_generator import whole_buffer
from nets.model import EncoderDecoder
import soundfile as sf
from os.path import join
import util.onehot as onehot
import config

WIDTH = 80
ENDPAD = 6
BUFPAD = 16
THICKNESS = 9

def _PAD(j, m):
    return ((j + m - 1) // m) * m

class bufmixer:
    z = numpy.linspace(0, 1, WIDTH)
    bufmixer0 = numpy.maximum(2*z - 1, 0)[:, None]
    bufmixer1 = 1 - numpy.abs(2*z - 1)[:, None]
    bufmixer2 = numpy.maximum(1 - 2*z, 0)[:, None]
    
    @staticmethod
    def mixer(buf):
        return numpy.concatenate([buf * bufmixer.bufmixer0,
                                  buf * bufmixer.bufmixer1,
                                  buf * bufmixer.bufmixer2], axis=2)

def _buffer_block(buffers):
    longest_buf = 0
    for buf in buffers:
        assert buf.shape[1:] == (WIDTH, 3)
        longest_buf = max(longest_buf, buf.shape[0])
    longest_buf = _PAD(longest_buf, BUFPAD)
    bufmatrix = numpy.zeros((len(buffers), longest_buf, WIDTH, 9))
    for i, buf in enumerate(buffers):
        bufmatrix[i, :buf.shape[0], :, :] = bufmixer.mixer(buf)
    return bufmatrix

def _trans_block(transs):
    longest_trans = 0
    for trans in transs:
        longest_trans = max(longest_trans, len(trans))
    longest_trans += 1 + ENDPAD
    shape = (len(transs), longest_trans, onehot.nchars)
    transmatrix = numpy.zeros(shape)
    transoffset = numpy.zeros(shape)
    for i, trans in enumerate(transs):
        trans = "@" + trans + "$" * (longest_trans - len(trans) - 1)
        for j, c in enumerate(trans):
            transmatrix[i, j, onehot.idx[c]] = 1
    transoffset[:, :-1, :] = transmatrix[:, 1:, :]
    transoffset[:, -1, onehot.idx["$"]] = 1
    return transmatrix, transoffset

def _create(get, start, end):
    bufs = []
    transs = []
    for i in range(0, end - start):
        trans, buf = get(start + i)
        bufs.append(buf)
        transs.append(trans)
    bufmatrix = _buffer_block(bufs)
    transmatrix, transoffset = _trans_block(transs)
    return (bufmatrix, transmatrix), transoffset

class WarpEvaluator:
    def __init__(self):
        pass

    def __call__(self, buf, src, dst):
        with tf.device("/cpu:0"):
            ret = tf.contrib.image.sparse_image_warp(buf, src, dst, num_boundary_points=1)[0]
            return ret.eval(session=tf.keras.backend.get_session())

warpevaluator = WarpEvaluator()

def warp(buf):
    batch, tau, height, channels = buf.shape
    W = min(20, round(tau/4))
    src = numpy.zeros((batch, 1, 2))
    dst = numpy.zeros((batch, 1, 2))
    src[:, 0, 1] = tau/2
    dst[:, 0, 1] = tau/2
    src[:, 0, 0] = W + (tau - 2*W) * numpy.random.random(size=batch)
    dst[:, 0, 0] = src[:, 0, 0] - W + 2*W*numpy.random.random(size=batch)
    buf[:] = warpevaluator(buf, src, dst)

def frequency_mask(buf, m):
    batch, tau, height, channels = buf.shape
    F = 27
    for i in range(m):
        f = numpy.ceil(F * numpy.random.random(size=batch)).astype(int)
        f0 = ((height - f) * numpy.random.random(size=batch)).astype(int)
        for j in range(batch):
            buf[j, :, f0[j]:f[j]+f0[j], :] = 0

def time_mask(buf, m):
    batch, tau, height, channels = buf.shape
    T = min(40, round(tau / 2))
    for i in range(m):
        t = numpy.ceil(T * numpy.random.random(size=batch)).astype(int)
        t0 = ((tau - t) * numpy.random.random(size=batch)).astype(int)
        for j in range(batch):
            buf[j, t0[j]:t[j]+t0[j], :, :] = 0

def specaugment(buf):
    #warp(buf)
    frequency_mask(buf, 2)
    time_mask(buf, 2)

"""
    longest_buf = ((longest_buf + BUFPAD - 1) // BUFPAD) * BUFPAD
    bufmatrix = numpy.zeros((len(bufs), longest_buf, WIDTH, 9))
    for i, buf in enumerate(bufs):
        bufmatrix[i, :buf.shape[0], :, :] = bufmixer.mixer(buf)
    longest_trans += 1 + ENDPAD
    transmatrix = numpy.zeros((len(transs), longest_trans, onehot.nchars))
    transoffset = numpy.zeros((len(transs), longest_trans, onehot.nchars))
    #transoffset = numpy.zeros_like(transmatrix)
    for i, trans in enumerate(transs):
        trans = "@" + trans + "$" * (longest_trans - len(trans) - 1)
        for j, c in enumerate(trans):
            transmatrix[i, j, onehot.idx[c]] = 1
    transoffset[:, :-1, :] = transmatrix[:, 1:, :]
    transoffset[:, -1, onehot.idx["$"]] = 1
    return (bufmatrix, transmatrix), transoffset
"""

class SequenceFromLibriSpeech(tf.keras.utils.Sequence):
    def __init__(self, dat, batchsize, get, drop_chars=False):
        self.data = dat
        self.batchsize = batchsize
        #self.wb = wholebuffer
        self.get = get
        self.dropchar = drop_chars
    
    def __len__(self):
        return (len(self.data) + self.batchsize - 1) // self.batchsize

    def __getitem__(self, idx):
        start = idx * self.batchsize
        end = min((idx + 1) * self.batchsize, len(self.data))
        #print("returning batch %d %d" % (start, end))
        retval = _create(self.get, start, end)
        if self.dropchar:
            buf = retval[0][0]
            specaugment(buf)
            transmatrix = retval[0][1]
            transmatrix = transmatrix * (0.9 - 0.1/30) + 0.1/30
            for i in range(transmatrix.shape[0]):
                for j in range(transmatrix.shape[1]):
                    if numpy.random.random() < 0.1:
                        dist = numpy.exp(numpy.random.normal(size=transmatrix.shape[2]))
                        transmatrix[i, j, :] = dist / numpy.sum(dist)
        return retval
        
class LibriSequence:
    def __init__(self):
        self.path = config.path
        self.ls = LibriSpeech()
        self.ls.load()
        self.batchsize = 4
    
    def sequence(self, type="train"):
        def get(ix):
            reader,book,i = self.ls.info[type][ix]
            file = self.ls.data[reader][book][i]
            buf, _ = sf.read(join(self.path, file['path']))
            trans = file['trans']
            wb = whole_buffer()
            wb.params.spectrum_range = config.librispeech_range
            return trans, wb.all(buf)
        return SequenceFromLibriSpeech(self.ls.info[type], self.batchsize, get, drop_chars=type == "train")

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

"""
def lrsche(epoch):
    rate = 4
    until = 9
    if epoch <= until:
        return 0.001 + (0.01 - 0.001) * epoch / until
        #return 0.001 + (0.01 - 0.001) * (epoch - 1) / (until - 1)
        # start at 0????
    elif epoch <= 17:
        return 0.01 * 0.5**((epoch - until) / rate)
    else:
        return lrsche(epoch - 18)
        """

from importlib import reload
from traceback import print_exc
def lrsche(epoch):
    while True:
        try:
            import lrsche
            reload(lrsche)
            return lrsche.lrsche(epoch)
        except:
            print_exc()
            print("")
            input("Hit return to try and load it again.")

def train(save="", epoch_=0):
    encdec = EncoderDecoder()
    spectrum = tf.keras.layers.Input((None, WIDTH, 9))
    transcript = tf.keras.layers.Input((None, len(onehot.chars)))
    decode = encdec(spectrum, transcript)
    #decode = tf.nn.softmax(decode)
    model = tf.keras.models.Model([spectrum, transcript], decode)
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0, nesterov=True),
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    samp = LibriSequence()
    checkp = tf.keras.callbacks.ModelCheckpoint(
            filepath = "checkpoints/weights.{epoch:04d}-{val_loss:.2f}.hdf5",
            save_weights_only = True
    )
    if save != "":
        model.load_weights(save)
    l = tf.keras.callbacks.LearningRateScheduler(lrsche, verbose=1)
    model.fit_generator(
        samp.sequence("train"),
        epochs = 100,
        verbose = 1,
        validation_data = samp.sequence("test"),
        initial_epoch = epoch_,
        workers = 4, shuffle=False, callbacks=[checkp, l]
    )
