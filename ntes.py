from nets.model import convblock
from util.sample import Sampler
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

sa = Sampler()
file, buf = sa.rand()
t = tf.convert_to_tensor(buf.reshape((1,) + buf.shape).astype(np.float32))
cb = convblock(64, 2)

print(cb(t))