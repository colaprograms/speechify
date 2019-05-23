from util.sample import Sampler
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

sa = Sampler()
file, buf = sa.rand()
t = tf.convert_to_tensor(buf.reshape((1,) + buf.shape).astype(np.float32))

import nets.model
de = nets.model.decoder()
en = nets.model.encoder()
t = en(t)
t = de(t, t)
print(t)
