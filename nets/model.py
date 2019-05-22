import tensorflow as tf
import numpy as np
#from tensorflow.keras.layers import * #Bidirectional, CuDNNLSTM, Conv2D, Dense, \
        #Embedding, Concatenate, LeakyReLU, BatchNormalization

def Conv2D(m, kernel_size, strides=1, **kwargs):
    args = {
        'padding': "same",
        'kernel_initializer': "he_normal",
        'use_bias': False
    }
    args.update(kwargs)
    return tf.keras.layers.Conv2D(m, kernel_size=3, strides=strides, **args)

# modified from https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/keras/layers/cudnn_recurrent.py
def bias_initializer_two(channels):
    """This bias initializer has more or less the effect of unit_forget_bias,
    but the starting bias is two, not just one.
    Hopefully this will help with very-long-term memory."""
    def f(_, *args, **kwargs):
        return array_ops.concat([
            tf.keras.initializers.Zeros((channels * 5,), *args, **kwargs),
            tf.keras.initializers.Constant(2)((channels,), *args, **kwargs),
            tf.keras.initializers.Zeros((channels * 2,), *args, **kwargs),
        ], axis=0)

def LSTM(channels, **kwargs):
    args = dict(
        return_sequences = True,
        bias_initializer = bias_initializer_two(channels)
    )
    args.update(kwargs)
    return tf.keras.layers.CuDNNLSTM(channels, **args)
     
from tensorflow.keras.layers import BatchNormalization, \
                            LeakyReLU, \
                            GlobalAveragePooling2D, \
                            Dense, \
                            Reshape, \
                            Dropout, \
                            Bidirectional
from tensorflow.keras import Sequential

class initial_state(tf.keras.layers.Layer):
    def __init__(self, units,
            bias_initializer = "zeros",
            bias_regularizer = None,
            bias_constraint = None):
        super(initial_state, self).__init__(**kwargs)
        self.units = units
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.input_spec = tf.keras.layers.InputSpec(min_ndim = 2)
    
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.bias = self.add_weight(
            shape = (self.units,),
            initializer = self.bias_initializer,
            name = "initial_state",
            regularizer = self.bias_regularizer,
            constraint = self.bias_constraint
        )
        self.input_spec = tf.keras.layers.InputSpec(min_ndim = 2, axes = {-1: input_dim})
        self.built = True
    
    def call(self, inputs):
        return tf.broadcast_to(inputs, self.compute_output_shape(tf.shape(inputs)))
    
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)
    
    def get_config(self):
        config = dict(
            units = self.units,
            bias_initializer = tf.keras.initializers.serialize(self.bias_initializer),
            bias_regularizer = tf.keras.initializers.serialize(self.bias_regularizer),
            bias_constraint = tf.keras.constraints.serialize(self.bias_constraint)
        )
        base_config = super(initial_state, self).get_config().copy()
        base_config.update(config)
        return base_config
            
class convblock(tf.keras.Model):
    def __init__(self, channels, stride=1):
        super(convblock, self).__init__()
        self.conv0 = Conv2D(channels, 1, stride)
        self.conv1 = Conv2D(channels, 3, stride)
        self.bn1 = BatchNormalization(axis=3)
        self.act1 = LeakyReLU()
        self.conv2 = Conv2D(channels, 3)
        
        self.squeeze = tf.keras.Sequential([
            GlobalAveragePooling2D(),
            Reshape((1, 1, channels)),
            Dense(channels // 16, activation='relu', kernel_initializer='he_normal'),
            Dense(channels, activation='sigmoid', kernel_initializer='he_normal')
        ])
        
        self.bn2 = BatchNormalization(axis=3)
        self.act2 = LeakyReLU()
    
    def _squeeze(self, out):
        squ = self.squeeze(out)
        return tf.keras.layers.multiply([squ, out])
    
    def call(self, x):
        shortcut = self.conv0(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self._squeeze(out)
        out = self.bn2(out)
        out = self.act2(out)
        return tf.keras.layers.add([shortcut, out])
        
class encoder(tf.keras.Model):
    def __init__(self):
        super(encoder, self).__init__()
        # Different parts of the mel spectrum act differently,
        # so we should probably add location data to
        # the starting image, like rotating between two phases
        # or adding a spinner.
        WIDTH = 160
        
        self.conv1 = convblock(16, 1)
        self.conv2 = convblock(16, 2)
        self.conv3 = convblock(64, 1)
        self.conv4 = convblock(64, 2)

        self.flatten_spectrogram = Reshape((-1, 64 * WIDTH // 4))
        
        def _lstm(size):
            lstm = Bidirectional(LSTM(size))
            var = initial_state(size)
            return lambda z: lstm(z, initial_state = [var(z), var(z)])
        
        def _pyra(size):
            # I'm not sure what the paper means by "projection layer."
            # Zhang et al has a thing where they staple two adjacent
            # frames together and project it down.
            # On the other hand, the Bidirectional layer also returns
            # twice the size by default, I think, so project all four
            # of those things down? Or two separate projections?
            def pyramids(zz):
                pads = tf.constant([[0, 0],
                                    [0, tf.floormod(tf.shape(zz)[0], 2)], # padded along height axis
                                    [0, 0]])
                zz = tf.pad(zz, pads, "CONSTANT")
                zz = tf.concat([zz[:, ::2, :], zz[:, 1::2, :]], -1)
                return zz
            proj = Dense(size)
            batc = BatchNormalization()
            def fn(buf):
                return batc(proj(pyramids(buf)))
            return fn
            
        self.lstm1 = _lstm(256)
        self.pyra1 = _pyra(256)
        self.drop1 = Dropout(0.1)
        
        self.lstm2 = _lstm(256)
        self.pyra2 = _pyra(256)
        self.drop2 = Dropout(0.1)
        
        self.lstm3 = _lstm(256)
        self.pyra3 = _pyra(256)
        self.drop3 = Dropout(0.1)
        
    def call(self, zz):
        for fn in [self.conv1, self.conv2, self.conv3, self.conv4]:
            zz = fn(zz)
        
        zz = self.flatten_spectrogram(zz)
        
        zz, _ = self.lstm1(zz)
        zz = self.pyra1(zz)
        zz = self.drop1(zz)
        
        zz, _ = self.lstm2(zz)
        zz = self.pyra2(zz)
        zz = self.drop2(zz)
        
        zz, _ = self.lstm3(zz)
        zz = self.pyra3(zz)
        zz = self.drop3(zz)
        
        # should i be remembering the state?
        return zz

    def initialize_hidden_state(self, bsiz):
        pass