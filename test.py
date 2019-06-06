import tensorflow as tf
import tensorflow.keras.backend as B
import numpy as np
import nets.model
import nets.main
import nets.decode
import config
import util.onehot as onehot

if __name__ == "__main__":
    if config.place == "home":
        print("Setting allow_growth")
        #print("Setting float16 and allow_growth")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        B.set_session(tf.Session(config=config))
    f = nets.decode.ModelDecoder()
    f.build()
    trans, buf = nets.decode._random_test_buf()
    transmatrix, transoffset = nets.main._trans_block([trans])
    if True:
        Z = tf.placeholder(np.float32, (None, None, 80, 9))
        tran = tf.placeholder(np.float32, (None, None, onehot.nchars))
        z = f.encdec(Z, tran)
        h = B.function([Z,tran], z)
        decodes = h([buf, transmatrix])
        out = ""
        out2 = ""
        for i in range(decodes.shape[1]):
            out += onehot.chars[np.argmax(transmatrix[0, i, :])]
            out2 += onehot.chars[np.argmax(decodes[0, i, :])]
        print(out)
        print(out2)
        print(np.mean([a == b for (a, b) in zip(out[1:], out2[:-1])]))
    """
    if True:
        for i in range(20):
            out2 = "@" + out2
            decodes = np.zeros((1, len(out2), 30))
            for i in range(len(out2)):
                decodes[-1, i, onehot.idx[out2[i]]] = 1
            decodes = h([buf, decodes])
            out2 = ""
            for i in range(decodes.shape[1]):
                out2 += onehot.chars[np.argmax(decodes[0, i, :])]
            print(out2)
    """

    if True:
        Z = tf.placeholder(np.float32, (None, None, 80, 9))
        z = f.encdec.enc(Z)
        h = B.function(Z, z)
        decoder = B.function(*f.encdec.dec.decode_one())

        speech_encode = h(buf)
        print(speech_encode.shape)
        z = lambda: np.zeros((1, 256), dtype=np.float32)
        last = ((z(), z()), (z(), z()))
        #last = [(np.zeros((1, 256), dtype=np.float32), np.zeros((1, 256), dtype=np.float32)) for j in range(2)]
        out = ""
        out2 = ""
        for i in range(transmatrix.shape[1]):
            out += onehot.chars[np.argmax(transmatrix[0, i, :])]
            decodes, last = decoder([transmatrix[:, i, :], speech_encode, last])
            print(trans[i] if i < len(trans) else "$", "".join(sorted((c for c in onehot.chars), key=lambda c: -decodes[0, onehot.idx[c]])))
            #out2 += onehot.chars[np.argmax(decodes[0, :])]
        print(out)
        print(out2)
        print(np.mean([a == b for (a, b) in zip(out[1:], out2[:-1])]))

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
