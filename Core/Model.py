import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.model_layers = []

    def add_layer(self, layer):
        self.model_layers.append(layer)

    def call(self, x: tf.Tensor, training: bool = True):
        for layer in self.model_layers:
            x = layer(x)
        return x