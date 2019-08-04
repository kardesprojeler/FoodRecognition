import tensorflow as tf


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.model_networks = {}
        self.model_layers = []

    def add_network(self, name=''):
        if name not in self.model_networks:
            self.model_networks[name] = []

    def add_layer(self, layer, *network_names):
        if len(self.model_networks) == 0:
            self.add_network("main")

        if len(network_names) > 0 and len(self.model_networks) > 0:
            raise Exception('Herhangi bir network bulunamadı!')
        elif len(network_names) == 0:
            self.model_networks["main"].append(layer)
        else:
            for network_name in network_names:
                if network_name in self.model_networks:
                    self.model_networks[network_name].append(layer)
                else:
                    raise Exception('Layerin ekleneceği network bulunamadı!\nLayer Name:{}'.format(layer.name))

    def call(self, x: tf.Tensor, training: bool = True):
        for network in self.model_networks:
            for layer in network:
                x = layer(x)
        return x

