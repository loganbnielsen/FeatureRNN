import log
import logging
logger = logging.getLogger('root')

import tensorflow as tf

from tensorflow.keras.layers import Dense

class FFN(tf.keras.Model):
    def __init__(self, layers, activ):
        """
            layers    = list of integers representing the number of nodes in each layer
            activ     = the activation func between each layer
        """
        super().__init__()
        self.ffn_layers = []
        for i, dim in enumerate(layers[:-1],start=1):
            self.ffn_layers.append( Dense(dim, activation=activ, name=f"layer{i}") )
        self.ffn_layers.append( Dense(layers[-1], name="output layer") )
    
    def call(self, X):
        for l in self.ffn_layers:
            X = l(X)
        return X