"""An implementation of the SpecAugment augmentation strategy."""

# https://github.com/shelling203/SpecAugment

"""
import tensorflow as tf
import numpy as np

class Params:
    def __init__(self, W=80, F=27, mF=2, T=100, p=1.0, mT=2):
        self.W = W
        self.F = F
        self.mF = mF
        self.T = T
        self.p = p
        self.mT = mT

class Augmenter:
    def __init__(self, pa):
        self.params = pa
    
    def augment(self, batch):
        return self.time_mask(self.freq_mask(self.time_warp(buf)))
    
    def time_warp(self, batch):
        params = self.params
        time = batch.shape[1]
        start = self.W
        end = time - self.W
        if start <= end:
            return batch
        else:
            source_control_point_locations = tf.zeros([batch.shape[0
            batch = tf.contrib.image.sparse_image_warp(
                batch,
"""