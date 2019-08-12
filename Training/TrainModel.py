from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf  # TF2

from absl import app
from Models.DenseNet import DenseNet
from Models.SimpleModel import SimpleModel
from Datas.Data import *
from Datas.Utils import *
from tensorflow.python import keras
from keras.utils import generic_utils

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
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.losses.Reduction.NONE)
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

    def train_step(self, dataset_inputs, augment, strategy):
        def step_fn(inputs):
            images, y_class, y_regr = inputs
            rpn_class_predict, rpn_regr_predict = self.model.call_networks(images, "rpn_class", "rpn_regr")

            rpn_class_loss = self.loss_function(y_class, rpn_class_predict, 1)
            rpn_regr_loss = self.loss_function(y_regr, rpn_regr_predict, 1)

            rpn_class_train_op = self.optimizer.minimize(rpn_class_loss)
            rpn_regr_train_op = self.optimizer.minimize(rpn_regr_loss)

            R = rpn_to_roi(rpn_class_predict, rpn_regr_predict, FasterRCNNConfig, 'tf', use_regr=True, overlap_thresh=0.7, max_boxes=300)
            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            class_mapping = get_class_mapping()
            X2, Y1, Y2, IouS = calc_iou(R, augment, FasterRCNNConfig, class_mapping)

            if X2 is not None:
                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)

                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []

                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []

                if FasterRCNNConfig.num_rois > 1:
                    if len(pos_samples) < FasterRCNNConfig.num_rois // 2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, FasterRCNNConfig.num_rois // 2,
                                                                replace=False).tolist()
                    try:
                        selected_neg_samples = np.random.choice(neg_samples,
                                                                FasterRCNNConfig.num_rois - len(selected_pos_samples),
                                                                replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples,
                                                                FasterRCNNConfig.num_rois - len(selected_pos_samples),
                                                                replace=True).tolist()

                    sel_samples = selected_pos_samples + selected_neg_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()
                    if np.random.randint(0, 2):
                        sel_samples = random.choice(neg_samples)
                    else:
                        sel_samples = random.choice(pos_samples)

                out_class_predict, out_regr_predict = self.model.call_networks(images, "out_class", "out_regr")
                out_class_loss = self.loss_function(X2[:, sel_samples, :], out_class_predict, 1)
                out_regr_loss = self.loss_function([Y1[:, sel_samples, :], Y2[:, sel_samples, :]], out_regr_predict, 1)

                out_class_train_op = self.optimizer.minimize(rpn_class_loss)
                out_regr_train_op = self.optimizer.minimize(rpn_regr_loss)

                self.progbar.update(0 + 1, [('rpn_cls', rpn_class_loss),
                                            ('rpn_regr', rpn_regr_loss),
                                            ('detector_cls', out_class_loss),
                                            ('detector_regr', out_regr_loss)])

                with tf.control_dependencies([rpn_class_train_op, rpn_regr_train_op, out_class_train_op, out_regr_train_op]):
                    return tf.identity(rpn_class_train_op + rpn_regr_train_op + out_class_train_op + out_regr_train_op)

        per_replica_losses = strategy.experimental_run_v2(
            step_fn, args=(dataset_inputs,))
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

    def training_loop(self, train_dist_dataset, augments, strategy):
        """Custom training loop.
        Args:
          train_dist_dataset: Training dataset created using strategy.
          strategy: Distribution strategy
        Returns:
          train_loss
        """

        def distributed_train_epoch(ds):
            total_loss = 0.0
            num_train_batches = 0.0
            per_replica_loss = strategy.experimental_run_v2(
                self.train_step, args=(ds, augments, strategy))
            total_loss += strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
            num_train_batches += 1

            return total_loss, num_train_batches

        template = 'Epoch: {}, Train Loss: {}, Train Accuracy: {}'
        train_total_loss = 0
        num_train_batches = 1
        self.progbar = generic_utils.Progbar(self.epochs)
        for epoch in range(self.epochs):
            for x in train_dist_dataset:
                train_total_loss, num_train_batches = self.train_step(x, augments, strategy)
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
        train_dist_dataset, augments = get_datasets(strategy, batch_size, model.get_img_output_length)

        train_obj = Train(epochs, False, model)
        print('Training ...')
        return train_obj.training_loop(train_dist_dataset, augments, strategy)


def train_model(model_name, argv):
    run_main(model_name, argv)
