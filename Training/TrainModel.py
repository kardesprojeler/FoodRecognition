from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf  # TF2

from absl import app
from Models.DenseNet import DenseNet
from Models.SimpleModel import SimpleModel
from Datas.Data import *
from tensorflow.python import keras


class Train(object):
    """Train class.
    Args:
    epochs: Number of epochs
    enable_function: If True, wraps the train_step and test_step in tf.function
    model: Densenet model.
    """

    def __init__(self, epochs, enable_function, model):
        self.epochs = epochs
        self.enable_function = enable_function
        self.autotune = tf.data.experimental.AUTOTUNE
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.optimizer = tf.train.GradientDescentOptimizer(0.001)
        self.training_loss = tf.keras.metrics.Mean("training_loss", dtype=tf.float32)
        self.training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            "training_accuracy", dtype=tf.float32)
        self.test_loss = tf.keras.metrics.Mean("test_loss", dtype=tf.float32)
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            "test_accuracy", dtype=tf.float32)
        self.test_acc_metric = keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy', dtype=tf.float32)
        self.model = model
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=model)
        pass

    def loss_function(self, real, pred, batch_size):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_) * 1. / batch_size

    def train_step(self, dist_inputs, strategy):
        def train_step(dist_inputs):
            def step_fn(inputs):
                images, labels = inputs
                rpn_class_predict, rpn_regr_predict = self.model.call_networks(images, "rpn_class", "rpn_regr")
                rpn_class_loss = self.loss_function(labels, rpn_class_predict, 1)
                rpn_regr_loss = self.loss_function(labels, rpn_regr_predict, 1)

                rpn_class_train_op = self.optimizer.minimize(rpn_class_loss)
                rpn_regr_train_op = self.optimizer.minimize(rpn_regr_loss)

                with tf.control_dependencies([rpn_class_train_op, rpn_regr_train_op]):
                    return tf.identity(rpn_class_train_op + rpn_regr_train_op)

            per_replica_losses = strategy.experimental_run_v2(
                step_fn, args=(dist_inputs,))
            mean_loss = strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            return mean_loss

    def test_step(self, images, labels, batch_size):
        """One test step.
        Args:
          inputs_test: tuple of input tensor, target tensor.
        Returns:
          Loss value so that it can be used with `tf.distribute.Strategy`.
        """

        logits = self.model(images)
        loss = self.loss_function(logits, labels, batch_size)
        self.test_loss.update_state(loss)
        self.test_accuracy.update_state(labels, logits)

    def training_loop(self, imgs, y_class, y_regr, augments, strategy):
        """Custom training loop.
        Args:
          train_dist_dataset: Training dataset created using strategy.
          strategy: Distribution strategy
        Returns:
          train_loss
        """
        @tf.function()
        def distributed_train_epoch(ds):
            total_loss = 0.0
            num_train_batches = 0.0
            for images, labels in ds:
                per_replica_loss = strategy.experimental_run_v2(
                    self.train_step, args=(images, labels, strategy))
                total_loss += strategy.reduce(
                    tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
                num_train_batches += 1
            return total_loss, num_train_batches

        if self.enable_function:
            distributed_train_epoch = tf.function(distributed_train_epoch)

        template = 'Epoch: {}, Train Loss: {}, Train Accuracy: {}'
        train_total_loss = 0
        num_train_batches = 1
        for epoch in range(self.epochs):
            train_total_loss, num_train_batches = distributed_train_epoch(train_dist_dataset)
            print(template.format(epoch, train_total_loss / num_train_batches, self.training_accuracy.result() * 100))

        return (train_total_loss / num_train_batches)


def run_main(model_name, argv):
    main(model_name, GeneralFlags.epoch.value)


def main(model_name, epochs):
    model = None
    batch_size = 1
    class_list = get_sinif_list()
    class_number = len(class_list)
    if model_name == 'SimpleModel':
        model = SimpleModel(SimpleModelFlags.pool_initial.value, SimpleModelFlags.init_filter.value,
                            SimpleModelFlags.stride.value, SimpleModelFlags.growth_rate.value,
                            SimpleModelFlags.image_height.value, SimpleModelFlags.image_width.value,
                            SimpleModelFlags.image_deep.value, SimpleModelFlags.batch_size.value,
                            SimpleModelFlags.save_path.value, class_number)
        batch_size = SimpleModelFlags.batch_size.value
        pass
    elif model_name == 'DenseNet':
        model = DenseNet(DenseNetFlags.mode.value, DenseNetFlags.growth_rate.value, DenseNetFlags.output_classes.value,
                         DenseNetFlags.depth_of_model.value, DenseNetFlags.num_of_blocks.value,
                         DenseNetFlags.num_layers_in_each_block.value, DenseNetFlags.data_format.value,
                         DenseNetFlags.bottleneck.value, DenseNetFlags.compression.value,
                         DenseNetFlags.weight_decay.value, DenseNetFlags.dropout_rate.value,
                         DenseNetFlags.pool_initial.value, DenseNetFlags.include_top.value)
        batch_size = DenseNetFlags.batch_size.value
        pass
    elif model_name == 'ResNet':
        pass

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():

        imgs, y_class, y_regr, augments = get_datasets(strategy, batch_size, model.get_img_output_length)

        train_obj = Train(epochs, False, model)
        print('Training ...')
        return train_obj.training_loop(imgs, y_class, y_regr, augments, strategy)


def train_model(model_name, argv):
    run_main(model_name, argv)
