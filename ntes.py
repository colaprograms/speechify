import tensorflow as tf
import numpy as np
import nets.model
import nets.main

if __name__ == "__main__":
    #tf.enable_eager_execution()
    nets.main.train()

"""sa = Sampler()
file, buf = sa.rand()
print(file)
t = tf.convert_to_tensor(buf.reshape((1,) + buf.shape).astype(np.float32))

import nets.model
de = nets.model.decoder()
en = nets.model.encoder()
t = en(t)
t = de(t, t)
print(t)
"""
