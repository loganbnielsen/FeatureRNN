import log
import logging
logger = logging.getLogger('root')

import tensorflow as tf

class GRUCell(tf.keras.layers.Layer):
    def __init__(self, dim_u, dim_x):
        """
            dim_x = 'state space' dim
            dim_u = input dim
        """
        super().__init__()
        self.dim_u, self.dim_x = dim_u, dim_x
        orthogonal = tf.initializers.orthogonal()
        self.Az, self.Ar, self.Ah = [ tf.Variable( orthogonal([1, dim_x, dim_x]) )
                                      for _ in range(3) ]
        self.Bz, self.Br, self.Bh = [ tf.Variable( orthogonal([1, dim_x, dim_u]) )
                                      for _ in range(3) ]
        self.bz, self.br, self.bh = [ tf.Variable( orthogonal([1, dim_x, 1]) )
                                      for _ in range(3) ]
        self.sig = tf.sigmoid
        self.tanh = tf.tanh

    @property
    def init_x(self):
        return tf.zeros([1,self.dim_x, 1])

    def __call__(self, u, x=None):
        """
            u (B x DIM_U x 1)
            x (B x DIM_X x 1)
            returns (B x DIM_X x 1)
        """
        logger.debug(f"u.shape={u.shape}; u={u}")
        logger.debug(f"x.shape={x.shape}; x={x}")
        x = x if tf.is_tensor(x) else self.init_x
        z     = self.sig(  self.Az @ x + self.Bz @ u + self.bz)
        r     = self.sig(  self.Ar @ x + self.Br @ u + self.br)
        x_hat = self.tanh( self.Ah @ (r*x) + self.Bh @ u + self.bh)
        return (1 - z) * x + z * x_hat 