import log
import logging
logger = logging.getLogger('root')

import tensorflow as tf

from FRNN import FRNN
from FFN import FFN

class Model(tf.keras.Model):
    def __init__(self, w, k, n, q, h, z, layers, activ_func):
        """
            FRNN:
                w = window length (number of periods)
                k = dimension of time (note: time2vec will make time-dim k+1)
                n = number of features at each point in time (without time features)
                q = number of queries (column in W)
                h = number of heads
                z = number of features to be extracted from the q*h results created by the attention heads
            FFN:
                layers    = list of integers representing the number of nodes in each layer
                activ     = the activation func between each layer
        """
        super().__init__()
        self.frnn = FRNN(w, k, n, q, h, z)
        self.ffn  = FFN(layers, activ_func)

        self.w, self.k, self.n, self.q, self.h, self.z, self.layer_dims, self.activ_func = \
            w, k, n, q, h, z, layers, activ_func

    def call(self, I, t, x=None):
        """
            I (B x W x N)
            t (B x W x 1) time vector
            x (B x Z x 1) hidden state

            return (B x 1), (B x 1 x Z)
        """
        logger.debug(f"x[t].shape = {x.shape}; x[t] = {x}")
        x = self.frnn(I,t,x) # (B x Z x 1)
        logger.debug(f"x[t+1].shape = {x.shape}; x[t+1] = {x}")
        y = self.ffn( tf.transpose(x, perm=[0,2,1]) )
        logger.debug(f"y[t+1].shape = {y.shape}; y[t+1] = {y}")
        return y, x
