import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import * #Bidirectional, CuDNNLSTM, Conv2D, Dense, \
        #Embedding, Concatenate, LeakyReLU, BatchNormalization

def conv33(m, s=1):
    return Conv2D(m, kernel_size=3, strides=s, padding="same", kernel_initializer="he_normal", use_bias=False)

def conv11(m, s):
    return Conv2D(m, (1, 1), padding="same", kernel_initializer="he_normal", use_bias=False, strides=s)

class convblock(tf.keras.Model):
    def __init__(self, channels, stride=1):
        super(convblock, self).__init__()
        self.conv0 = conv11(channels, stride)
        self.conv1 = conv33(channels, stride)
        self.bn1 = BatchNormalization(axis=3)
        self.act1 = LeakyReLU()
        self.conv2 = conv33(channels)
        
        self.squeeze = tf.keras.Sequential([
            GlobalAveragePooling2D(),
            Reshape((1, 1, channels)),
            Dense(channels // 16, activation='relu', kernel_initializer='he_normal', use_bias=False),
            Dense(channels, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
        ])
        self.bn2 = BatchNormalization(axis=3)
        self.act2 = LeakyReLU()
        
    def call(self, x):
        shortcut = self.conv0(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        squ = self.squeeze(out)
        out = Multiply()([squ, out])
        out = self.bn2(out)
        out = self.act2(out)
        return Add()([shortcut, out])
        
class encoder(tf.keras.Model):
    def __init__(self):
        super(encoder, self).__init__()
        # Different parts of the mel spectrum act differently,
        # so we should probably add location data to
        # the starting image, like rotating between two phases
        # or adding a spinner.
        self.layer1 = convblock(16, 1)
        self.layer2 = convblock(16, 2)
        self.layer3 = convblock(64, 1)
        self.layer4 = convblock(64, 2)

        def _lstm(size):
            lstm = Bidirectional(LSTM(size, return_sequences = True))
            # I'm not sure what the paper means by "projection layer."
            # Zhang et al has a thing where they staple two adjacent
            # frames together and project it down.
            # On the other hand, the Bidirectional layer also returns
            # twice the size by default, I think, so project all four
            # of those things down? Or two separate projections?
            proj = Dense(size)
            batc = BatchNormalization()
            return lstm, proj, batc

        self.lstm1 = _lstm(256)
        self.lstm2 = _lstm(256)
        self.lstm3 = _lstm(256)

    def call(self, zz, hidden):
        zz = self.layer1(zz)
        zz = self.layer2(zz)
        zz = self.layer3(zz)
        zz = self.layer4(zz)
        
        zz, _ = self.lstm1(zz, initial_state = hidden1)
        zz, state = self.lstm2(zz, initial_state = hidden2)
        return zz, state

    def initialize_hidden_state(self, bsiz):
        pass