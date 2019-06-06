import tensorflow as tf
import numpy as np
#from tensorflow.keras.layers import * #Bidirectional, CuDNNLSTM, Conv2D, Dense, \
        #Embedding, Concatenate, LeakyReLU, BatchNormalization
from util.onehot import nchars
        
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
    return tf.constant_initializer(
            [0.0] * (channels*5) + [2.0] * channels + [0.0] * (channels*2)
    )
    return f
def bias_initializer_two_cell(channels):
    """This bias initializer has more or less the effect of unit_forget_bias,
    but the starting bias is two, not just one.
    Hopefully this will help with very-long-term memory."""
    return tf.constant_initializer(
            [0.0] * (channels) + [2.0] * channels + [0.0] * (channels*2)
    )
    return f

def LSTM(channels, **kwargs):
    args = dict(
        return_sequences = True,
        unit_forget_bias = False, bias_initializer = bias_initializer_two(channels)
    )
    args.update(kwargs)
    return tf.keras.layers.CuDNNLSTM(channels, **args)

def LSTMCell(channels, **kwargs):
    args = dict(
        recurrent_activation = "sigmoid",
        unit_forget_bias = False, bias_initializer = bias_initializer_two_cell(channels)
    )
    args.update(kwargs)
    return tf.keras.layers.LSTMCell(channels, **args)
 
from tensorflow.keras.layers import BatchNormalization, \
                            LeakyReLU, \
                            GlobalAveragePooling2D, \
                            Dense, \
                            Reshape, \
                            Dropout, \
                            Bidirectional
from tensorflow.keras import Sequential

def _lstm(size):
    return Bidirectional(LSTM(size))

def _pyra(size):
    # I'm not sure what the paper means by "projection layer."
    # Zhang et al has a thing where they staple two adjacent
    # frames together and project it down.
    # On the other hand, the Bidirectional layer also returns
    # twice the size by default, I think, so project all four
    # of those things down? Or two separate projections?
    def pyramids(zz):
        # assuming tf.shape(zz)[1] is zero mod BUFPAD
        zz = tf.concat([zz[:, ::2, :], zz[:, 1::2, :]], -1)
        return zz
    proj = Dense(size)
    batc = BatchNormalization()
    def fn(buf):
        return batc(proj(pyramids(buf)))
    return fn

def whatever_norm(z):
    mean, variance = tf.nn.moments(z, [z.shape.ndims - 1], keep_dims = True)
    return tf.nn.batch_normalization(z, mean, variance, None, None, variance_epsilon=1e-4)
    #return tf.contrib.layers.layer_norm(z, center=False, scale=False)

class initial_state(tf.keras.layers.Layer):
    def __init__(self, units,
            bias_initializer = "zeros",
            bias_regularizer = None,
            bias_constraint = None):
        super(initial_state, self).__init__()
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
        return self.bias
    
    def compute_output_shape(self, input_shape):
        assert len(input_shape) >= 2
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
            
class convblock(tf.keras.layers.Layer):
    def __init__(self, channels, stride=1):
        super(convblock, self).__init__()
        self.conv0 = Conv2D(channels, 1, stride)
        self.conv1 = Conv2D(channels, 3, stride)
        #self.bn1 = BatchNormalization(axis=3)
        self.act1 = LeakyReLU()
        self.conv2 = Conv2D(channels, 3)
        
        self.squeeze = tf.keras.Sequential([
            GlobalAveragePooling2D(),
            Reshape((1, 1, channels)),
            Dense(channels // 16, activation='relu', kernel_initializer='he_normal'),
            Dense(channels, activation='sigmoid', kernel_initializer='he_normal')
        ])
        
        #self.bn2 = BatchNormalization(axis=3)
        self.act2 = LeakyReLU()
    
    def _squeeze(self, out):
        squ = self.squeeze(out)
        return tf.keras.layers.multiply([squ, out])
    
    def call(self, x):
        shortcut = self.conv0(x)
        out = self.conv1(x)
        out = whatever_norm(out)
        #out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self._squeeze(out)
        #out = self.bn2(out)
        out = self.act2(out)
        # atashi iya ne
        return whatever_norm(tf.keras.layers.add([shortcut, out]))
        
class encoder(tf.keras.layers.Layer):
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

        self.lstm1 = _lstm(512)
        self.drop1 = Dropout(0.1)
        
        self.lstm2 = _lstm(512)
        self.drop2 = Dropout(0.1)
        
        self.lstm3 = _lstm(512)
        self.drop3 = Dropout(0.1)
        
    def call(self, zz):
        for fn in [self.conv1, self.conv2, self.conv3, self.conv4]:
            zz = fn(zz)
        
        zz = self.flatten_spectrogram(zz)
        
        zz = self.lstm1(zz)
        zz = whatever_norm(zz)
        zz = self.drop1(zz)
        
        zz = self.lstm2(zz)
        zz = whatever_norm(zz)
        zz = self.drop2(zz)
        
        zz = self.lstm3(zz)
        zz = whatever_norm(zz)
        zz = self.drop3(zz)
        
        # should i be remembering the state?
        return zz

    def initialize_hidden_state(self, bsiz):
        pass

class AttentionCell(tf.keras.layers.Layer):
    def __init__(self, units):
        super(AttentionCell, self).__init__()
        self.units = units
        self.Wa = Dense(self.units)
        self.va = Dense(1)
        self.cell = LSTMCell(units)
        self.state_size = [units, units]
    
    def build(self, input_shape):
        self.built = True
        
    def call(self, inputs, states, constants):
        speech_encode, encodestate = constants
        print("states:", states)
        state = self.Wa(states[0])
        state = tf.expand_dims(state, axis=1)
        attentions = self.va(tf.tanh(state + encodestate))
        attention_logits = tf.squeeze(attentions, axis=2)
        attention_weights = tf.nn.softmax(attention_logits)
        context = tf.einsum('ai,aij->aj',
                attention_weights, speech_encode)
        lstm_in = tf.concat([inputs, context], axis=1)
        lstmout, hiddenstate = self.cell(lstm_in, states)
        #print(hiddenstate)
        return lstmout, hiddenstate
            
class attend(tf.keras.layers.Layer):
    def __init__(self, units):
        super(attend, self).__init__()
        self.units = units
        """Bahdanau attention:
        e_{ij} = v_a^T tanh(W_a s_{i-1} + U_a h_j)
        \alpha_{ij} = exp(e_{ij}) / \sum_j exp(e_{ij})
        """
        self.Ua = Dense(self.units)
        self.cell = tf.keras.layers.RNN(AttentionCell(units),
            return_sequences = True)

    def call(self, inputs):
        secrets, speech_encode = inputs
        encodestate = self.Ua(speech_encode)
        return self.cell(secrets, constants=(speech_encode, encodestate))

    def get_encode_state(self, speech_encode):
        return self.Ua(speech_encode)

    def call_one(self, secrets, last_state, speech_encode):
        encodestate = self.Ua(speech_encode)
        return self.cell.cell.call(secrets, last_state, constants=(speech_encode, encodestate))
        #return self.cell.cell.call(secrets, last_state, constants)
        
class decoder(tf.keras.layers.Layer):
    def __init__(self):
        tf.keras.layers.Layer.__init__(self)
        self.units = 256
        self.embedding = Dense(self.units)
        self.attends1 = attend(256)
        self.attends2 = attend(256)
        self.map1 = Dense(256)
        self.map2 = Dense(256)
        self.distrib = Dense(nchars)
        
    def call(self, inputs):
        trans, speech_encode = inputs
        secrets = self.embedding(trans)
        print(tf.shape(secrets))
        print(tf.shape(speech_encode))
        out = self.attends1([secrets, speech_encode])
        out = whatever_norm(out)
        out = self.attends2([out, speech_encode])
        out = whatever_norm(out)
        out = self.map1(out)
        out = whatever_norm(out)
        out = self.map2(out)
        out = whatever_norm(out)
        out = self.distrib(out)
        out = tf.nn.softmax(out)
        return out

    def get_encode_state(self, speech_encode):
        return (self.attends1.get_encode_state(speech_encode),
                self.attends2.get_encode_state(speech_encode))

    def prepare_encode(self):
        placeholder = lambda *z: tf.placeholder(tf.float32, z)
        speech_encode = placeholder(1, None, None)
        return speech_encode, self.get_encode_state(speech_encode)

    def decode_one(self):
        placeholder = lambda *z: tf.placeholder(tf.float32, z)
        trans = placeholder(1, None)
        speech_encode = placeholder(1, None, None)
        last1 = [placeholder(1, 256), placeholder(1, 256)]
        last2 = [placeholder(1, 256), placeholder(1, 256)]
        inputs = [trans, speech_encode, (last1, last2)]

        secrets = self.embedding(trans)
        out, last1 = self.attends1.call_one(secrets, last1, speech_encode)
        out = whatever_norm(out)
        out, last2 = self.attends2.call_one(out, last2, speech_encode)
        out = whatever_norm(out)
        out = self.map1(out)
        out = whatever_norm(out)
        out = self.map2(out)
        out = whatever_norm(out)
        out = self.distrib(out)
        out = tf.nn.softmax(out)
        outputs = [out, (last1, last2)]
        return inputs, outputs

class EncoderDecoder(tf.keras.Model):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.enc = encoder()
        self.dec = decoder()
    
    def call(self, spectrum, transcript):
        speech_encode = self.enc(spectrum)
        return self.dec([transcript, speech_encode])
    
    def loss(self, transcript, decode):
        pass
