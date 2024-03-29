import pprint
import sys
import time
from optparse import OptionParser
from Datas.Data import *
from Datas.Utils import *
from keras.utils import generic_utils
import tensorflow as tf


parser = OptionParser()

parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
                  default="pascal_voc")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.",
                  default=32)
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.",
                  default='resnet50')
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).",
                  action="store_true", default=False)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).",
                  action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90",
                  help="Augment with 90 degree rotations in training. (Default=false).",
                  action="store_true", default=False)
parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=10)
parser.add_option("--config_filename", dest="config_filename",
                  help="Location to store all the metadata related to the training (to be used when testing).",
                  default="config.pickle")
parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.",
                  default='./model_frcnn.hdf5')
parser.add_option("--input_weight_path", dest="input_weight_path",
                  help="Input path for weights. If not specified, will try to load default weights provided by keras.")

(options, args) = parser.parse_args()

# pass the settings from the command line, and persist them in the config object

from Models.ResNet import ResNet as nn

class_mapping = get_class_mapping()

all_imgs, classes_count = get_data(r'C:\Users\Durkan\Desktop\train_images.txt')

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

model = nn()

data_gen_train = get_anchor_gt(all_imgs, classes_count, FasterRCNNConfig, model.get_img_output_length)

input_shape_img = (GeneralFlags.train_image_height.value, GeneralFlags.train_image_width.value, 3)

img_input = tf.keras.layers.Input(shape=input_shape_img)
roi_input = tf.keras.layers.Input(shape=(None, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = model.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(FasterRCNNConfig.anchor_box_scales) * len(FasterRCNNConfig.anchor_box_ratios)
rpn = model.rpn(shared_layers, num_anchors)

classifier = model.classifier(shared_layers, roi_input, FasterRCNNConfig.num_rois,
                              nb_classes=len(class_mapping), trainable=True)

model_rpn = tf.keras.models.Model(img_input, rpn[:2])
model_classifier = tf.keras.models.Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = tf.keras.models.Model([img_input, roi_input], rpn[:2] + classifier)

if os.path.exists(FasterRCNNConfig.model_path):
    try:
        print('loading weights from {}'.format(FasterRCNNConfig.model_path))
        model_rpn.load_weights(FasterRCNNConfig.model_path, by_name=True)
        model_classifier.load_weights(FasterRCNNConfig.model_path, by_name=True)
    except:
        print('Could not load pretrained model weights. Weights can be found in the keras application folder '
              'https://github.com/fchollet/keras/tree/master/keras/applications')


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
optimizer_classifier = tf.keras.optimizers.Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss=[rpn_loss_cls(num_anchors), rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier,
                         loss=[class_loss_cls, class_loss_regr(len(class_mapping) - 1)],
                         metrics={'dense_class_{}'.format(len(class_mapping)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

epoch_length = 50
num_epochs = 5
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf
vis = True

for epoch_num in range(num_epochs):

    progbar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

    for i in range(epoch_length):
        try:
            if len(rpn_accuracy_rpn_monitor) == epoch_length and FasterRCNNConfig.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                    mean_overlapping_bboxes, epoch_length))
                if mean_overlapping_bboxes == 0:
                    print(
                        'RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
            X, Y, img_data = next(data_gen_train)

            loss_rpn = model_rpn.train_on_batch(X, Y)

            P_rpn = model_rpn.predict_on_batch(X)

            R = rpn_to_roi(P_rpn[0], P_rpn[1], FasterRCNNConfig, 'tf', use_regr=True, overlap_thresh=0.7, max_boxes=300)
            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            X2, Y1, Y2, IouS = calc_iou(R, img_data, FasterRCNNConfig, class_mapping)

            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue

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

            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

            if FasterRCNNConfig.num_rois > 1:
                if len(pos_samples) < FasterRCNNConfig.num_rois // 2:
                    selected_pos_samples = pos_samples.tolist()
                else:
                    selected_pos_samples = np.random.choice(pos_samples, FasterRCNNConfig.num_rois // 2, replace=False).tolist()
                try:
                    selected_neg_samples = np.random.choice(neg_samples, FasterRCNNConfig.num_rois - len(selected_pos_samples),
                                                            replace=False).tolist()
                except:
                    selected_neg_samples = np.random.choice(neg_samples, FasterRCNNConfig.num_rois - len(selected_pos_samples),
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

            loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                         [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]

            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]

            progbar.update(iter_num + 1, [('rpn_cls', losses[iter_num, 0]), ('rpn_regr', losses[iter_num, 1]),
                                          ('detector_cls', losses[iter_num, 2]),
                                          ('detector_regr', losses[iter_num, 3])])

            iter_num += 1
            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                if FasterRCNNConfig.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                        mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))

                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()

                if FasterRCNNConfig.verbose:
                    print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                best_loss = curr_loss

        except Exception as e:
            print('Exception: {}'.format(e))
            continue
    print("Saving Weights")
    model_all.save_weights(FasterRCNNConfig.model_path)