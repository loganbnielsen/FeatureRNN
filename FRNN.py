import log
import logging
logger = logging.getLogger('root')

import tensorflow as tf

from Time2Vec import Time2Vec
from Attention import MultiHeadAttention
from GRU import GRUCell

class FRNN(tf.keras.Model):
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
        self.W = tf.Variable( tf.random.uniform([1,1,w]) )
        self.gru = GRUCell(dim_u=z, dim_x=z)
        # self.batchnorm = tf.keras.layers.BatchNormalization(axis=-2)

    @property
    def init_hidden(self):
        return tf.zeros([1,self.z,1])

    def call(self, I, t, x=None):
        """
            I (B x W x N)
            t (B x W x 1) time vector
            x (B x Z x 1) hidden state

            return (B x Z x 1)
        """
        x = x if tf.is_tensor(x) else self.init_hidden
        T = self.time2vec(t)
        logger.debug(f"T.shape={T.shape}; T={T}")
        X = tf.concat([I,T],-1) # B x W x N+K+1
        logger.debug(f"X.shape={X.shape}; X={X}")
        F = self.mha(X) # B x W x Z
        logger.debug(f"F.shape={F.shape}; F={F}")
        f = tf.transpose((self.W @ F), perm=[0,2,1]) # B x Z x 1
        logger.debug(f"f.shape={f.shape}; f={f}")
        r = self.gru(f, x) # B x Z x 1
        logger.debug(f"r.shape={r.shape}; r={r}")
        x = f+r
        x = x/tf.linalg.norm(x) # B x Z x 1
        # x = self.batchnorm(f + r) # B x Z x 1
        logger.debug(f"x.shape={x.shape}; x={x}")
        return x





