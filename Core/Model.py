import tensorflow as tf
import copy

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.model_networks = {}

    def add_network(self, *network_names):
        for name in network_names:
            if name not in self.model_networks:
                self.model_networks[name] = []

    def add_layer(self, layer, add_all_networks=True, *network_names):
        if len(self.model_networks) == 0:
            self.add_network("main")

        if len(self.model_networks) == 1 and "main" in self.model_networks:
            self.model_networks["main"].append(layer)
        else:
            if add_all_networks:
                for network_key in self.model_networks:
                    self.model_networks[network_key].append(layer)
            else:
                for network_name in list(network_names):
                    if network_name in self.model_networks:
                        self.model_networks[network_name].append(layer)
                    else:
                        raise Exception('Layerin ekleneceği network bulunamadı!\nLayer Name:{}'.format(layer.name))

    def call(self, x: tf.Tensor, training: bool = True):
        for network in self.model_networks:
            for layer in network:
                x = layer(x)
        return x

    def call_networks(self, input, *network_nemes):
        result = []
        for name in network_nemes:
            if name in self.model_networks:
                out = copy.deepcopy(input)
                for layer in self.model_networks[name]:
                    out = layer(out)
                result.append(out)

        return result




