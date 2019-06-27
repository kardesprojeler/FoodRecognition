from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf  # TF2

assert tf.__version__.startswith('2')
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
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.5)
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

    def train_step(self, image_batch, label_batch, model_name):
        with tf.GradientTape() as tape:
            loss = None
            predictions = None

            if model_name == 'SimpleModel':
                predictions = self.model(image_batch)
                loss = self.loss_function(label_batch, predictions, SimpleModelFlags.batch_size.value)
                pass
            elif model_name == 'DenseNet':
                predictions = self.model(image_batch, training=True)
                loss = self.loss_function(label_batch, predictions, DenseNetFlags.batch_size.value)
                pass
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.training_accuracy.update_state(label_batch, predictions)
        self.training_loss(loss)
        return loss

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

    def training_loop(self, train_dist_dataset, test_dist_dataset, is_there_test, model_name, strategy):
        """Custom training and testing loop.
        Args:
          train_dist_dataset: Training dataset created using strategy.
          test_dist_dataset: Testing dataset created using strategy.
          strategy: Distribution strategy
        Returns:
          train_loss, test_loss
        """
        @tf.function()
        def distributed_train_epoch(ds):
            total_loss = 0.0
            num_train_batches = 0.0
            for images, labels in ds:
                per_replica_loss = strategy.experimental_run_v2(
                    self.train_step, args=(images, labels, model_name))
                total_loss += strategy.reduce(
                    tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
                num_train_batches += 1
            return total_loss, num_train_batches

        @tf.function()
        def distributed_test_epoch(ds):
            for images, labels in ds:
                strategy.experimental_run_v2(
                    self.test_step, args=(images, labels, model_name))
            return self.test_loss.result()

        if self.enable_function:
            distributed_train_epoch = tf.function(distributed_train_epoch)
            distributed_test_epoch = tf.function(distributed_test_epoch)

        template = 'Epoch: {}, Train Loss: {}, Train Accuracy: {}'
        train_total_loss = 0
        num_train_batches = 1
        test_total_loss = 0
        for epoch in range(self.epochs):
            train_total_loss, num_train_batches = distributed_train_epoch(train_dist_dataset)
            if is_there_test:
                test_total_loss = distributed_test_epoch(test_dist_dataset)
            print(template.format(epoch, train_total_loss / num_train_batches, self.training_accuracy.result() * 100))

        return (train_total_loss / num_train_batches, test_total_loss)


def run_main(model_name, argv):
    main(model_name, GeneralFlags.epoch.value, GeneralFlags.enable_function.value, GeneralFlags.train_mode.value)


def main(model_name, epochs, enable_function, train_mode):
    model = None
    batch_size = 1
    if model_name == 'SimpleModel':
        model = SimpleModel(SimpleModelFlags.pool_initial.value, SimpleModelFlags.init_filter.value,
                            SimpleModelFlags.stride.value, SimpleModelFlags.growth_rate.value,
                            SimpleModelFlags.image_height.value, SimpleModelFlags.image_width.value,
                            SimpleModelFlags.image_deep.value, SimpleModelFlags.batch_size.value,
                            SimpleModelFlags.save_path.value)
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

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():

        train_ds, test_ds, is_there_test = get_datasets(strategy, batch_size)

        train_obj = Train(epochs, False, model)
        print('Training ...')
        return train_obj.training_loop(train_ds, test_ds, is_there_test, model_name, strategy)


def get_datasets(strategy, batch_size):
    x_train, y_train = read_train_images(200, 200)
    x_test, y_test = read_test_images(200, 200)

    # Numpy defaults to dtype=float64; TF defaults to float32. Stick with float32.
    x_train, x_test = x_train / np.float32(255), x_test / np.float32(255)
    y_train, y_test = y_train.astype(np.int64), y_test.astype(np.int64)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(batch_size)
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

    return train_dist_dataset, test_dist_dataset, len(x_test) > 0


def train_model():
    app.run(run_main)
