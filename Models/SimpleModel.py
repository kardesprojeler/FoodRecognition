from Datas.Data import *
from Datas.Utils import *
from Core.Model import Model


class SimpleModel(Model):
    def __init__(self, pool_initial, init_filters, stride, grow_rate, image_height, image_width,
                 image_deep, batch_size, save_path):
        super(SimpleModel, self).__init__()
        self.l2 = tf.keras.regularizers.l2
        self.pool_initial = pool_initial
        self.init_filters = init_filters
        self.stride = stride
        self.grow_rate = grow_rate
        self.image_height = image_height
        self.image_width = image_width
        self.image_deep = image_deep
        self.batch_size = batch_size
        self.save_path = save_path
        self.prepare_layer()

    def is_model_prepared(self):
            return self.sess is not None

    def prepare_layer(self):
        self.num_class = len(get_sinif_list())
        self.add_layer(self.conv_layer(24))
        self.add_layer(tf.keras.layers.BatchNormalization())
        self.add_layer(self.conv_layer(48))
        self.add_layer(tf.keras.layers.BatchNormalization())
        self.add_layer(self.conv_layer(24))
        self.add_layer(tf.keras.layers.BatchNormalization())

        self.add_layer(tf.keras.layers.Flatten())

        self.add_layer(self.dense_layer(64, activation='relu'))
        self.add_layer(self.dense_layer(32, activation='relu'))

        self.add_layer(self.dense_layer(2, activation='softmax'))

    def conv_layer(self, num_filters):
        return tf.keras.layers.Conv2D(num_filters,
                                      self.init_filters,
                                      strides=self.stride,
                                      padding="same",
                                      use_bias=False,
                                      data_format='channels_last',
                                      kernel_initializer="he_normal",
                                      kernel_regularizer=self.l2(1e-4))
        pass

    def dense_layer(self, size=24, activation='relu'):
        return tf.keras.layers.Dense(units=size, activation=activation)
        pass


class RPN(Model):
    def __init__(self):
        super(RPN, self).__init__()
        self.prepare_layer()

    def prepare_layer(self):
        self.add_layer(tf.keras.layers.ZeroPadding2D((3, 3)))
        self.add_layer(tf.keras.layers.Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', trainable=True))
        self.add_layer(FixedBatchNormalization(axis=3, name='bn_conv1'))
        self.add_layer(tf.keras.layers.Activation('relu1'))
        self.add_layer(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)))
        self.add_layer(tf.keras.layers.Convolution2D(64, (1, 1), strides=(1, 1), name='conv2', trainable=True))
        self.add_layer(FixedBatchNormalization(axis=3, name='bn_conv2'))
        self.add_layer(tf.keras.layers.Activation('relu2'))
        self.add_layer(tf.keras.layers.Convolution2D(64, (3, 3), padding='same', name='conv3', trainable=True))
        self.add_layer(FixedBatchNormalization(axis=3, name='bn_conv3'))
        self.add_layer(tf.keras.layers.Activation('relu3'))
        self.add_layer(tf.keras.layers.Convolution2D(256, (1, 1), name='conv4', trainable=True))
        self.add_layer(FixedBatchNormalization(axis=3, name='bn_conv4'))
        self.add_layer(tf.keras.layers.Activation('relu4'))
        self.add_layer(tf.keras.layers.Convolution2D(64, (1, 1), name='conv5', trainable=True))
        self.add_layer(FixedBatchNormalization(axis=3, name='bn_conv4'))
        self.add_layer(tf.keras.layers.Activation('relu'))