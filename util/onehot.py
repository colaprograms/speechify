import numpy as np

chars = " ABCDEFGHIJKLMNOPQRSTUVWXYZ'@$"
nchars = len(chars)
# @ is start token, $ is end token
char = {}
idx = {}
for i, c in enumerate(chars):
    char[i] = c
    idx[c] = i

def text_to_onehot(tx):
    buf = np.zeros((len(tx), NLETTER), dtype=np.float32)
    for i in range(len(tx)):
        buf[i, idx[tx[i]]] = 1
    return buf
