from Datas.Data import *
from Datas.Utils import *
from Core.Model import Model


class SimpleModel(Model):
    def __init__(self, pool_initial, init_filters, stride, grow_rate, image_height, image_width,
                 image_deep, batch_size, save_path, num_class):
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
        self.num_class = num_class
        self.prepare_model_networks()

    def get_img_output_length(self, width, height):
        def get_output_length(input_length):
            # zero_pad
            input_length += 6
            # apply 4 strided convolutions
            filter_sizes = [7, 3, 1, 1]
            stride = 2
            for filter_size in filter_sizes:
                input_length = (input_length - filter_size + stride) // stride
            return input_length

        return get_output_length(width), get_output_length(height)

    def prepare_model_networks(self):
        rpn_class = "rpn_class"
        rpn_regr = "rpn_regr"
        out_class = "out_class"
        out_regr = "out_regr"

        self.add_network(rpn_class, rpn_regr, out_class, out_regr)
        self.add_base_layers()
        self.add_rpn_layers(rpn_class, rpn_regr)
        self.add_classifier_layers(out_class, out_regr)

    def is_model_prepared(self):
            return self.sess is not None

    def add_base_layers(self):
        self.add_layer(tf.keras.layers.ZeroPadding2D((3, 3)))
        self.add_layer(tf.keras.layers.Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', trainable=True))
        self.add_layer(FixedBatchNormalization(axis=3, name='bn_conv1'))
        self.add_layer(tf.keras.layers.Activation('relu'))
        self.add_layer(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)))
        self.add_layer(tf.keras.layers.Convolution2D(64, (1, 1), strides=(1, 1), name='conv2', trainable=True))
        self.add_layer(FixedBatchNormalization(axis=3, name='bn_conv2'))
        self.add_layer(tf.keras.layers.Activation('relu'))
        self.add_layer(tf.keras.layers.Convolution2D(64, (3, 3), padding='same', name='conv3', trainable=True))
        self.add_layer(FixedBatchNormalization(axis=3, name='bn_conv3'))
        self.add_layer(tf.keras.layers.Activation('relu'))
        self.add_layer(tf.keras.layers.Convolution2D(256, (1, 1), name='conv4', trainable=True))
        self.add_layer(FixedBatchNormalization(axis=3, name='bn_conv4'))
        self.add_layer(tf.keras.layers.Activation('relu'))
        self.add_layer(tf.keras.layers.Convolution2D(64, (1, 1), name='conv5', trainable=True))
        self.add_layer(FixedBatchNormalization(axis=3, name='bn_conv4'))
        self.add_layer(tf.keras.layers.Activation('relu'))

    def add_rpn_layers(self, rpn_class, rpn_regr):
        self.add_layer(tf.keras.layers.Convolution2D(512, (3, 3), padding='same', activation='relu',
                                                     kernel_initializer='normal', name='rpn_conv1'),
                       False, rpn_class, rpn_regr)

        self.add_layer(tf.keras.layers.Convolution2D(SimpleModelFlags.num_anchors.value, (1, 1), activation='sigmoid',
                                                     kernel_initializer='uniform', name='rpn_out_class'), False, rpn_class)
        self.add_layer(tf.keras.layers.Convolution2D(SimpleModelFlags.num_anchors.value * 4, (1, 1), activation='linear',
                                                     kernel_initializer='zero', name='rpn_out_regress'), False, rpn_regr)

    def add_classifier_layers(self, out_class, out_regr):
        pooling_regions = 14
        input_shape = (SimpleModelFlags.num_rois.value, 14, 14, 1024)

        self.add_layer(RoiPoolingConv(pooling_regions, SimpleModelFlags.num_rois.value), False, out_class, out_regr)

        self.add_layer(tf.keras.layers.TimeDistributed(tf.keras.layers.Convolution2D(512, (1, 1), strides=(2, 2),
                                                                                     trainable=True,
                                                                                     kernel_initializer='normal'),
                                                       input_shape=input_shape, name='out_distributed1'), False, out_class, out_regr)
        self.add_layer(tf.keras.layers.TimeDistributed(FixedBatchNormalization(axis=3), name='out_fixedbatchnorm1'), False, out_class, out_regr)
        self.add_layer(tf.keras.layers.Activation('relu'), False, out_class, out_regr)

        self.add_layer(tf.keras.layers.TimeDistributed(tf.keras.layers.Convolution2D(512, (3, 3),
                                                                                     padding='same',trainable=True,
                                                                                     kernel_initializer='normal'),
                                                       name='out_distributed2'), False, out_class, out_regr)
        self.add_layer(tf.keras.layers.TimeDistributed(FixedBatchNormalization(axis=3), name='out_fixedbatchnorm2'), False, out_class, out_regr)
        self.add_layer(tf.keras.layers.Activation('relu'), False, out_class, out_regr)

        self.add_layer(tf.keras.layers.TimeDistributed(tf.keras.layers.Convolution2D(2048, (1, 1),
                                                                                     kernel_initializer='normal'),
                                                       name='out_distributed3', trainable=True), False, out_class, out_regr)
        self.add_layer(tf.keras.layers.TimeDistributed(FixedBatchNormalization(axis=3), name='out_fixedbatchnorm3'), out_class, out_regr)
        self.add_layer(tf.keras.layers.Activation('relu'), False, out_class, out_regr)

        self.add_layer(tf.keras.layers.TimeDistributed(tf.keras.layers.Convolution2D(512, (1, 1), trainable=True,
                                                                                     kernel_initializer='normal'),
                                                       name='out_distributed4'), False, out_class, out_regr)
        self.add_layer(tf.keras.layers.TimeDistributed(FixedBatchNormalization(axis=3), name='out_fixedbatchnorm4'), False, out_class, out_regr)
        self.add_layer(tf.keras.layers.Activation('relu'), False, out_class, out_regr)

        self.add_layer(tf.keras.layers.TimeDistributed(tf.keras.layers.Convolution2D(512, (3, 3), trainable=True,
                                                                                     kernel_initializer='normal',
                                                                                     padding='same'),
                                                       name='out_distributed5'), False, out_class, out_regr)
        self.add_layer(tf.keras.layers.TimeDistributed(FixedBatchNormalization(axis=3), name='out_fixedbatchnorm5'), False, out_class, out_regr)
        self.add_layer(tf.keras.layers.Activation('relu'), False, out_class, out_regr)

        self.add_layer(tf.keras.layers.TimeDistributed(tf.keras.layers.Convolution2D(2048, (1, 1), trainable=True,
                                                                                     kernel_initializer='normal'),
                                                       name='out_distributed6'), False, out_class, out_regr)
        self.add_layer(tf.keras.layers.TimeDistributed(FixedBatchNormalization(axis=3), name='out_fixedbatchnorm6'), False, out_class, out_regr)
        self.add_layer(tf.keras.layers.Activation('relu'), False, out_class, out_regr)

        self.add_layer(tf.keras.layers.TimeDistributed(tf.keras.layers.AveragePooling2D((7, 7)), name='avg_pool'), False, out_class, out_regr)

        self.add_layer(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()), False, out_class, out_regr)

        self.add_layer(tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.num_class, activation='softmax', kernel_initializer='zero'),
            name='dense_class_{}'.format(self.num_class)), False, out_class)

        # note: no regression target for bg class
        self.add_layer(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4 * (self.num_class - 1), activation='linear',
                                                                             kernel_initializer='zero'),
                                                       name='dense_regress_{}'.format(self.num_class)), False, out_regr)