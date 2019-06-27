from __future__ import print_function
from __future__ import absolute_import

from Core.Model import Model
from keras import backend as K
from Datas.Utils import RoiPoolingConv
from Datas.Utils import FixedBatchNormalization
import tensorflow as tf


class ResNet(Model):
    def __init__(self):
        super(ResNet, self).__init__()

    def get_weight_path(self):
        if K.image_dim_ordering() == 'th':
            return 'resnet50_weights_th_dim_ordering_th_kernels_notop.h5'
        else:
            return 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'

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

    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):
        nb_filter1, nb_filter2, nb_filter3 = filters
        bn_axis = 3

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = tf.keras.layers.Convolution2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a',
                                          trainable=trainable)(input_tensor)

        x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                                          name=conv_name_base + '2b',
                                          trainable=trainable)(x)
        x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
        x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = tf.keras.layers.Convolution2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1',
                                                 trainable=trainable)(
            input_tensor)
        shortcut = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def identity_block(self, input_tensor, kernel_size, filters, stage, block, trainable=True):
        nb_filter1, nb_filter2, nb_filter3 = filters

        bn_axis = 3

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = tf.keras.layers.Convolution2D(nb_filter1, (1, 1), name=conv_name_base + '2a', trainable=trainable)(
            input_tensor)
        x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                                          name=conv_name_base + '2b',
                                          trainable=trainable)(x)

        x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
        x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = tf.keras.layers.Add()([x, input_tensor])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def identity_block_td(self, input_tensor, kernel_size, filters, stage, block, trainable=True):
        # identity block time distributed
        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_dim_ordering() == 'tf':
            bn_axis = 3
        else:
            bn_axis = 1

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Convolution2D(nb_filter1, (1, 1), trainable=trainable,
                                                                          kernel_initializer='normal'),
                                            name=conv_name_base + '2a')(input_tensor)
        x = tf.keras.layers.TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Convolution2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable,
                                          kernel_initializer='normal',
                                          padding='same'), name=conv_name_base + '2b')(x)
        x = tf.keras.layers.TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Convolution2D(nb_filter3, (1, 1), trainable=trainable,
                                                                          kernel_initializer='normal'),
                                            name=conv_name_base + '2c')(x)

        x = tf.keras.layers.TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

        x = tf.keras.layers.Add()([x, input_tensor])
        x = tf.keras.layers.Activation('relu')(x)

        return x

    def conv_block_td(self, input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):

        # conv block time distributed

        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_dim_ordering() == 'tf':
            bn_axis = 3
        else:
            bn_axis = 1

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Convolution2D(nb_filter1, (1, 1), strides=strides,
                                          trainable=trainable, kernel_initializer='normal'),
            input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)

        x = tf.keras.layers.TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Convolution2D(nb_filter2, (kernel_size, kernel_size),
                                                                          padding='same', trainable=trainable,
                                                                          kernel_initializer='normal'),
                                            name=conv_name_base + '2b')(x)

        x = tf.keras.layers.TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Convolution2D(nb_filter3, (1, 1), kernel_initializer='normal'),
                                            name=conv_name_base + '2c',
                            trainable=trainable)(x)
        x = tf.keras.layers.TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

        shortcut = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Convolution2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'),
            name=conv_name_base + '1')(input_tensor)
        shortcut = tf.keras.layers.TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '1')(shortcut)

        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def nn_base(self, input_tensor=None, trainable=False):

        # Determine proper input shape
        if K.image_dim_ordering() == 'th':
            input_shape = (3, None, None)
        else:
            input_shape = (None, None, 3)

        if input_tensor is None:
            img_input = tf.keras.layers.Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        if K.image_dim_ordering() == 'tf':
            bn_axis = 3
        else:
            bn_axis = 1

        x = tf.keras.layers.ZeroPadding2D((3, 3))(img_input)

        x = tf.keras.layers.Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', trainable=trainable)(x)
        x = FixedBatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.conv_block(input_tensor=x, kernel_size=3, filters=[64, 64, 256],
                            stage=2, block='a', strides=(1, 1), trainable=trainable)

        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b', trainable=trainable)
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c', trainable=trainable)

        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a', trainable=trainable)
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b', trainable=trainable)
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c', trainable=trainable)
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d', trainable=trainable)

        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a', trainable=trainable)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b', trainable=trainable)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c', trainable=trainable)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d', trainable=trainable)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e', trainable=trainable)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f', trainable=trainable)

        return x

    def classifier_layers(self, x, input_shape, trainable=False):

        # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
        # (hence a smaller stride in the region that follows the ROI pool)
        x = self.conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(2, 2),
                               trainable=trainable)

        x = self.identity_block_td(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
        x = self.identity_block_td(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.AveragePooling2D((7, 7)), name='avg_pool')(x)
        return x

    def rpn(self, base_layers, num_anchors):

        x = tf.keras.layers.Convolution2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal',
                          name='rpn_conv1')(base_layers)

        x_class = tf.keras.layers.Convolution2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform',
                                name='rpn_out_class')(x)
        x_regr = tf.keras.layers.Convolution2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero',
                               name='rpn_out_regress')(x)

        return [x_class, x_regr, base_layers]

    def classifier(self, base_layers, input_rois, num_rois, nb_classes=21, trainable=False):

        # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
        pooling_regions = 14
        input_shape = (num_rois, 14, 14, 1024)

        out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
        out = self.classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

        out = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(out)

        out_class = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(nb_classes, activation='softmax',
                                                                          kernel_initializer='zero'),
                                                    name='dense_class_{}'.format(nb_classes))(out)
        # note: no regression target for bg class
        out_regr = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4 * (nb_classes - 1), activation='linear',
                                                                         kernel_initializer='zero'),
                                                   name='dense_regress_{}'.format(nb_classes))(out)
        return [out_class, out_regr]