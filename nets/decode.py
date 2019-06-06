import tensorflow as tf
import numpy
from util.data import LibriSpeech
from util.spectrogram_generator import whole_buffer
import soundfile as sf
from os.path import join
import util.onehot as onehot
import config
from nets.model import EncoderDecoder

from nets.main import _buffer_block, _trans_block, WIDTH, ENDPAD

def _buf_from_file(fn):
    buf, _ = sf.read(fn)
    wb = whole_buffer()
    wb.params.spectrum_range = config.librispeech_range
    return wb.all(buf)

def _random_test_buf():
    ls = LibriSpeech()
    ls.load()
    file = ls.uniform_test()
    trans = file['trans']
    fn = join(config.path, file['path'])
    buf = _buf_from_file(fn)
    buf = _buffer_block([buf])
    return trans, buf

class ModelDecoder:
    def __init__(self):
        pass

    def build(self, save="models/weights-02-test.hdf5"):
        self.encdec = EncoderDecoder()
        self.enc = self.encdec.enc
        self.dec = self.encdec.dec
        spectrum = tf.keras.layers.Input((None, WIDTH, 9))
        transcript = tf.keras.layers.Input((None, len(onehot.chars)))
        decode = self.encdec(spectrum, transcript)
        self.model = tf.keras.models.Model([spectrum, transcript], decode)
        self.model.load_weights(save)
