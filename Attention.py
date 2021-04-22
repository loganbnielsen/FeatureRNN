import log
import logging
logger = logging.getLogger('root')

import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, n, q, h, z):
        """
            n = number of features (embedding dimension for words in NLP context)
            q = number of columns in the <Query, Key, Value> matrices
            h = number of heads
            z = number of features to be extracted from the q*h results created from the attention heads
        """
        super().__init__()
        self.heads = [AttentionHead(n,q) for _ in range(h)]
        self.W0 = tf.Variable(
                        tf.initializers.GlorotUniform()([1, q*h, z]),
                        shape = [1, q*h, z])
    
    def call(self, X):
        Z = tf.concat([h(X)  for h in self.heads], -1)
        return Z @ self.W0

class AttentionHead(tf.keras.layers.Layer):
    def __init__(self, n, q):
        """
            n = number of features (embedding dimension for words in NLP context)
            q = number of columns in the <Query, Key, Value> matrices


            Notes:
            softmax(  Q K.T / sqrt(dk) ) V
                = softmax( w (W_Q W_K.T) w.T / sqrt(d_k)) w * W_v
                = softmax( w M w.T / sqrt(d_k)) w * W_v
                ==> M = W_Q @ W_K.T (to save on computation)
        """
        super().__init__()
        self.n, self.q = n, q
        self.sqrt_q = q
        self.M = tf.Variable(
                      tf.initializers.GlorotUniform()(shape=[1,n,n]),
                    # tf.initializers.orthogonal()([1,n,n]),
                    shape = [1,n,n])
        self.W_v = tf.Variable(
                      tf.initializers.GlorotUniform()([1,n,q]),
                    # tf.initializers.orthogonal()([1,n,q]),
                    shape = [1,n,q])
        self.softmax = tf.nn.softmax

    def call(self, X):
        return self.softmax( (X @ self.M @ tf.transpose(X,[0,2,1]))/ self.sqrt_q ) @ X @ self.W_v
