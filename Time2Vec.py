import log
import logging
logger = logging.getLogger('root')

import tensorflow as tf

class Time2Vec(tf.keras.layers.Layer):
    def __init__(self, k):
        """
            Time2Vec(t)[i] = {
                  w_i * t + b_i   if i= 0
                F(w_i * t + b_i)  if 1 <= i <= k
            }
            Note: Size will be k+1
        """
        super().__init__()
        assert k > 0, "dimension must be greater than 0"
        self.k = k

        self.b0 = tf.Variable(
            tf.random.uniform([1, 1, 1], 0, 1),
        )
        self.b = tf.Variable(
            tf.random.uniform([1, 1, k], 0, 1)
        )
        self.w0 = tf.Variable(
            tf.random.uniform([1, 1, 1], 0, int(1e-5))
        )
        self.w = tf.Variable(
            tf.random.uniform([1, 1, k], 0, 1)
        )
        self.F = tf.sin

    def __call__(self, t):
        """
            t: (Batch x Window x 1) 
            returns: (Batch x Window x k+1)
        """
        T = tf.concat([self.w0 * t + self.b0,
                       self.F(self.w * t + self.b)], -1)
        return T