import log
import logging
logger = logging.getLogger('root')

import tensorflow as tf

from Time2Vec import Time2Vec
from Attention import MultiHeadAttention

class TimeAttention(tf.keras.Model):
    def __init__(self, w, k, n, q, h, z):
        """
            w = window length (number of periods)
            k = dimension of time (note: time2vec will make time-dim k+1)
            n = number of features at each point in time (without time features)
            q = number of queries (column in W)
            h = number of heads
            z = number of features to be extracted from the q*h results created by the attention heads
        """
        super().__init__()
        self.w, self.k, self.n, self.q, self.h, self.z = w, k, n, q, h, z
        self.time2vec = Time2Vec(k)
        self.mha = MultiHeadAttention(n+k+1, q, h, z)
        self.W  = tf.Variable( tf.random.uniform([1,1,w]) )
        self.C  = tf.Variable( tf.initializers.orthogonal()([1,z,1]) )
    
    def call(self, I, t):
        """
            I (B x W x N)
            t (B x W x 1) time vector
        """
        T = self.time2vec(t)
        X = tf.concat( [I,T], -1) # B x W x N+K+1
        F = self.mha(X) # B x W x Z
        out = self.W @ F @ self.C #  B x 1 x 1
        return out
