import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Bidirectional, CuDNNLSTM, Conv2D, Dense, \
        Embedding, Concatenate, LeakyReLU, BatchNormalization

class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        # Different parts of the mel spectrum act differently,
        # so we should probably add location data to
        # the starting image, like rotating between two phases
        # or adding a spinner.
        self.layer1 = "convolution 3x3"
        self.layer2 = "convolution 3x3"

        def _lstm(size):
            lstm = Bidirectional(LSTM(size, return_sequences = True))
            # I'm not sure what the paper means by "projection layer."
            # Zhang et al has a thing where they staple two adjacent
            # frames together and project it down.
            # On the other hand, the Bidirectional layer also returns
            # twice the size by default, I think, so project all four
            # of those things down? Or two separate projections?
            proj = Dense(size * 2, size)
            batc = BatchNormalization()
            return lstm, proj, batc

        self.lstm1 = _lstm(256)
        self.lstm2 = _lstm(256)
        self.lstm3 = _lstm(256)

    def call(self, zz, hidden):
        zz = self.conv1(zz)
        zz = self.conv2(zz)
        zz, _ = self.lstm1(zz, initial_state = hidden1)
        zz, state = self.lstm2(zz, initial_state = hidden2)
        return zz, state

    def initialize_hidden_state(self, bsiz):
