import tensorflow as tf
import numpy as np
import nets.model
import nets.main
import os
import glob

def last():
    ff = [(c, os.path.getmtime(c)) for c in glob.glob("checkpoints/*.hdf5")]
    if len(ff) == 0:
        return None
    else:
        return max(ff, key=lambda z: z[1])[0]

def train(f=""):
    nets.main.train(f) # ("weights.0002-1.84.hdf5")

def yesno(prompt):
    yn = "loop"
    while yn != "y" and yn != "n":
        yn = input(prompt).lower()
    return yn

if __name__ == "__main__":
    #tf.enable_eager_execution()
    #nets.main.train()
    lf = last()
    if lf is not None:
        print("Last checkpoint found:", lf)
        yn = yesno("Continue from this checkpoint? [yn] ")
        if yn == "y":
            train(lf)
        else:
            yn = yesno("Start training from scratch? [yn] ")
            if yn == "y":
                train()

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
