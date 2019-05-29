import tensorflow as tf
import numpy as np
import nets.model
import nets.main
import os
import glob
import config

def last():
    ff = [(c, os.path.getmtime(c)) for c in glob.glob("checkpoints/*.hdf5")]
    if len(ff) == 0:
        return None
    else:
        return max(ff, key=lambda z: z[1])[0]

def train(f, epoch):
    nets.main.train(f, epoch or 0) # ("weights.0002-1.84.hdf5")

def yesno(prompt):
    yn = "loop"
    while yn != "y" and yn != "n":
        yn = input(prompt).lower()
    return yn

if __name__ == "__main__":
    if config.place == "home":
        print("Setting allow_growth")
        #print("Setting float16 and allow_growth")
        #tf.keras.backend.set_floatx("float16")
        #tf.keras.backend.set_epsilon(1e-4)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.keras.backend.set_session(tf.Session(config=config))
    #tf.enable_eager_execution()
    #nets.main.train()
    lf = last()
    epoch = None
    if lf is not None:
        print("Last checkpoint found:", lf)
        yn = yesno("Continue from this checkpoint? [yn] ")
        if yn == "y":
            while epoch is None:
                what = input("Epoch? ")
                try:
                    whatint = int(what)
                    if whatint >= 0:
                        epoch = whatint
                except ValueError:
                    pass
                if epoch is None:
                    print(what, "is an invalid number of epochs.")

        if yn == "n":
            lf = None
            input("Hit return to train from scratch.")
    train(lf or "", epoch)

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
