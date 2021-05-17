import numpy as np

from tensorflow.python.keras import layers

class LinearNNModelTf(keras.Model):
    def __init__(self):
        super(network,self).__init__()
        self.layer = layers.Dense(1 , activation='relu')
        
    def call(self,x):
        y_pred = self.layer_1(x)
        return y_pred