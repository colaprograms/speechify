import tensorflow as tf
import numpy as np

DTYPE = np.float32
def zeros(shape):
    return np.zeros(shape, dtype=DTYPE)

"""Model structure:

Encoder:
    - Three layers of convolution
    - Three BLSTM

Decoder:
    attention
    LSTM"""

class TextEncoder:
    def __init__(self):
        "rar"

    def encode(self, text):
        buf = np.zeros((len(text), self.nletters), dtype=DTYPE)
def model():
    model = tf.keras.models.
