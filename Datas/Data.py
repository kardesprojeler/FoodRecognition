from __future__ import absolute_import
import os
import shutil
from pyodbc import connect
from tkinter import messagebox, filedialog
import numpy as np
import tensorflow as tf
import wx
from PIL import Image as pilimage
from absl import flags
from enum import Enum
import cv2
import math
import random
import copy
import threading
import itertools

FLAGS = flags.FLAGS

conn_str = (
    r'DRIVER={SQL Server};'
    r'SERVER=LAPTOP-1CAUHSG4;'
    r'DATABASE=YemekTanima;'
    r'Trusted_Connection=True;'
    )

cnxn = connect(conn_str)
cursor = cnxn.cursor()


def get_sinif_list():
    array = []
    cursor.execute("select id, sinifname, foldername, fiyat, doviz = case when dovizref = 1 then 'TL' else 'USD' end "
                   "from tbl_01_01_sinif")
    row = cursor.fetchone()
    while row:
        sinif = DataSinif()
        sinif.id = row[0]
        sinif.sinifname = row[1]
        sinif.foldername = row[2]
        sinif.fiyat = row[3]
        sinif.doviz = row[4]
        array.append(sinif)
        row = cursor.fetchone()
    return array


def read_train_images(heigh, width):
    siniflist = get_sinif_list()
    images = []
    labels = []
    for sinif in siniflist:
        path = os.path.join(os.getcwd(), "images",  sinif.foldername)
        for filename in os.listdir(path):
            file_content = cv2.imread(os.path.join(path, filename))
            if file_content is not None:
                im = cv2.resize(file_content, dsize=(heigh, width), interpolation=cv2.INTER_CUBIC)
                images.append(im)
                labels.append(siniflist.index(sinif))
            pass
        pass
    images, labels = random_batch(len(images), images, labels)
    return np.array(images), np.array(labels)


def read_test_images(heigh, width):
    siniflist = get_sinif_list()
    images = []
    labels = []
    for sinif in siniflist:
        cursor.execute('select id, foldername, filename from tbl_01_01_testimage a where id = ?', sinif.id)
        row = cursor.fetchone()
        while row:
            path = os.path.join(os.getcwd(), "images", row[1], row[2])
            file_content = cv2.imread(path)
            if file_content is not None:
                im = cv2.resize(file_content, dsize=(heigh, width), interpolation=cv2.INTER_CUBIC)
                images.append(im)
                labels.append(siniflist.index(sinif))
            row = cursor.fetchone()
        pass
    return np.array(images), np.array(labels)


def get_data_from_file(file_path):
    all_imgs = {}
    classes_count = {}

    with open(file_path, 'r', encoding="utf-8") as f:

        print('Parsing annotation files')

        for line in f:
            line_split = line.strip().split(',')
            (filename, x1, y1, x2, y2, class_id) = line_split

            if class_id not in classes_count:
                classes_count[class_id] = 1
            else:
                classes_count[class_id] += 1

            if filename not in all_imgs:
                all_imgs[filename] = {}

                img = cv2.imread(filename)
                (rows, cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []
            all_imgs[filename]['bboxes'].append(
                {'class_id': int(class_id), 'x1': float(x1), 'x2': float(x2), 'y1': float(y1), 'y2': float(y2)})

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        if 'bg' not in classes_count:
            classes_count["bg"] = 0

        return all_data, classes_count


def update_class(id, sinifname, foldername, fiyat, doviz):
    cursor.execute("select * from tbl_01_01_sinif where (id = ?)", id)
    row = cursor.fetchone()
    if row:
        cursor.execute("update tbl_01_01_sinif set sinifname = ?, foldername = ?, fiyat = ?, dovizref = ? where id = ?",
                       sinifname, foldername, fiyat, doviz, id)

    else:
        cursor.execute("insert into tbl_01_01_sinif(sinifname, foldername, fiyat, dovizref) " +
                       "values(?, ?, ?, ?)", sinifname, foldername, fiyat, doviz)
        cnxn.commit()


def delete_class(id):
    cursor.execute('delete tbl_01_01_sinif where id = ? ', id)
    cnxn.commit()
    pass

def add_training_file():
    src = filedialog.askopenfile()
    dest = filedialog.askdirectory(initialdir=r'C:\Users\BULUT\Documents\GitHub\YemekTanima\images')
    if not os.path.exists(os.path.join(dest, os.path.split(src)[1])):
        shutil.copy(src.name, dest)
        pass
    else:
        wx.MessageBox('Bu dosya mevcut!', 'Attention', wx.OK | wx.ICON_WARNING)
    pass


def add_test_image(parent, label_number):
    if label_number != None:
        with wx.FileDialog(None, 'Open', r'C:\Users\BULUT\Documents\GitHub\YemekTanima\images\Pilav',
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return  # the user changed their mind

            # save the current contents in the file
            pathname = fileDialog.GetPath()
            dest = r'C:\Users\BULUT\Documents\GitHub\YemekTanima\images\test_images'
            if not os.path.exists(os.path.join(dest, os.path.split(pathname)[1])):
                shutil.copy(pathname, dest)
                cursor.execute('insert into tbl_01_01_testimage(labelnumber, foldername, filename) '
                                    'values(?, ?, ?)', label_number, dest, os.path.split(pathname)[1])
                cnxn.commit()
                pass
            else:
                messagebox.showinfo('Bu dosya mevcut')
                pass
    else:
        messagebox.showinfo('Lütfen sınıf seçiniz')


def pre_process_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
    return image


def read_image(width, height):
    with wx.FileDialog(None, 'Open', r'C:\Users\BULUT\Documents\GitHub\YemekTanima\images',
                       style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:

        if fileDialog.ShowModal() == wx.ID_CANCEL:
            return  # the user changed their mind

        # save the current contents in the file
        pathname = fileDialog.GetPath()

        file_content = pilimage.open(pathname)
        im = file_content.resize((width, height), pilimage.ANTIALIAS)
        im = [np.array(im)]
        im = np.array(im).reshape((-1, width, height, 3))

        return im


def random_batch(batch_size, images, labels, append_preprocess=False):
    batch_images = []
    batch_labels = []
    for i in range(batch_size):
        random_number = np.random.randint(0, images.__len__())
        batch_labels.append(labels[random_number])
        img = images[random_number]

        if append_preprocess:
            pre_process_image(img)

        batch_images.append(img)
    return batch_images, batch_labels


def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def iou(a, b):
    # a and b should be (x1,y1,x2,y2)

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)


def get_new_img_size(width, height, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = img_min_side
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = img_min_side

    return resized_width, resized_height


class SampleSelector:
    def __init__(self, class_count):
        # ignore classes that have zero samples
        self.classes = [b for b in class_count.keys() if class_count[b] > 0]
        self.class_cycle = itertools.cycle(self.classes)
        self.curr_class = next(self.class_cycle)

    def skip_sample_for_balanced_class(self, img_data):

        class_in_img = False

        for bbox in img_data['bboxes']:

            cls_id = bbox['class_id']

            if cls_id == self.curr_class:
                class_in_img = True
                self.curr_class = next(self.class_cycle)
                break

        if class_in_img:
            return False
        else:
            return True


def calc_rpn(C, img_data, width, height, resized_width, resized_height, img_length_calc_function):
    downscale = float(C.rpn_stride)
    anchor_sizes = C.anchor_box_scales
    anchor_ratios = C.anchor_box_ratios
    num_anchors = len(anchor_sizes) * len(anchor_ratios)

    # calculate the output map size based on the network architecture

    (output_width, output_height) = img_length_calc_function(resized_width, resized_height)

    n_anchratios = len(anchor_ratios)

    # initialise empty output objectives
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

    num_bboxes = len(img_data['bboxes'])

    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    best_anchor_for_bbox = -1 * np.ones((num_bboxes, 4)).astype(int)
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

    # get the GT box coordinates, and resize to account for image resizing
    gta = np.zeros((num_bboxes, 4))
    for bbox_num, bbox in enumerate(img_data['bboxes']):
        # get the GT box coordinates, and resize to account for image resizing
        gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
        gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
        gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
        gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))

    # rpn ground truth

    for anchor_size_idx in range(len(anchor_sizes)):
        for anchor_ratio_idx in range(n_anchratios):
            anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
            anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]

            for ix in range(output_width):
                # x-coordinates of the current anchor box
                x1_anc = downscale * (ix + 0.5) - anchor_x / 2
                x2_anc = downscale * (ix + 0.5) + anchor_x / 2

                # ignore boxes that go across image boundaries
                if x1_anc < 0 or x2_anc > resized_width:
                    continue

                for jy in range(output_height):

                    # y-coordinates of the current anchor box
                    y1_anc = downscale * (jy + 0.5) - anchor_y / 2
                    y2_anc = downscale * (jy + 0.5) + anchor_y / 2

                    # ignore boxes that go across image boundaries
                    if y1_anc < 0 or y2_anc > resized_height:
                        continue

                    # bbox_type indicates whether an anchor should be a target
                    bbox_type = 'neg'

                    # this is the best IOU for the (x,y) coord and the current anchor
                    # note that this is different from the best IOU for a GT bbox
                    best_iou_for_loc = 0.0

                    for bbox_num in range(num_bboxes):

                        # get IOU of the current GT box and the current anchor box
                        curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]],
                                       [x1_anc, y1_anc, x2_anc, y2_anc])
                        # calculate the regression targets if they will be needed
                        if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
                            cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
                            cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
                            cxa = (x1_anc + x2_anc) / 2.0
                            cya = (y1_anc + y2_anc) / 2.0

                            tx = (cx - cxa) / (x2_anc - x1_anc)
                            ty = (cy - cya) / (y2_anc - y1_anc)
                            tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
                            th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))

                        if img_data['bboxes'][bbox_num]['class_id'] != 0:

                            # all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
                            if curr_iou > best_iou_for_bbox[bbox_num]:
                                best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                                best_iou_for_bbox[bbox_num] = curr_iou
                                best_x_for_bbox[bbox_num, :] = [x1_anc, x2_anc, y1_anc, y2_anc]
                                best_dx_for_bbox[bbox_num, :] = [tx, ty, tw, th]

                            # we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
                            if curr_iou > C.rpn_max_overlap:
                                bbox_type = 'pos'
                                num_anchors_for_bbox[bbox_num] += 1
                                # we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
                                if curr_iou > best_iou_for_loc:
                                    best_iou_for_loc = curr_iou
                                    best_regr = (tx, ty, tw, th)

                            # if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
                            if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
                                # gray zone between neg and pos
                                if bbox_type != 'pos':
                                    bbox_type = 'neutral'

                    # turn on or off outputs depending on IOUs
                    if bbox_type == 'neg':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'neutral':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'pos':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
                        y_rpn_regr[jy, ix, start:start + 4] = best_regr

    # we ensure that every bbox has at least one positive RPN region

    for idx in range(num_anchors_for_bbox.shape[0]):
        if num_anchors_for_bbox[idx] == 0:
            # no box with an IOU greater than zero ...
            if best_anchor_for_bbox[idx, 0] == -1:
                continue
            y_is_box_valid[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], best_anchor_for_bbox[
                    idx, 2] + n_anchratios *
                best_anchor_for_bbox[idx, 3]] = 1
            y_rpn_overlap[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], best_anchor_for_bbox[
                    idx, 2] + n_anchratios *
                best_anchor_for_bbox[idx, 3]] = 1
            start = 4 * (best_anchor_for_bbox[idx, 2] + n_anchratios * best_anchor_for_bbox[idx, 3])
            y_rpn_regr[
            best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], start:start + 4] = best_dx_for_bbox[idx, :]

    y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

    y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

    num_pos = len(pos_locs[0])

    # one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
    # regions. We also limit it to 256 regions.
    num_regions = 256

    if len(pos_locs[0]) > num_regions / 2:
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions / 2)
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = num_regions / 2

    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

    return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


def get_class_mapping(add_bg=True):
    class_mapping = {}
    for i, sinif in enumerate(get_sinif_list()):
        class_mapping[sinif.id] = {}
        class_mapping[sinif.id]["order"] = i
        class_mapping[sinif.id]["class_name"] = sinif.sinifname

    if add_bg:
        class_mapping[0] = {}
        class_mapping[0]["order"] = len(class_mapping) - 1
        class_mapping[0]["class_name"] = "bg"

    return class_mapping

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


def get_anchor_gt(all_img_data, class_count, C, img_length_calc_function, backend, mode='train'):
    # The following line is not useful with Python 3.5, it is kept for the legacy
    # all_img_data = sorted(all_img_data)

    sample_selector = SampleSelector(class_count)

    while True:
        if mode == 'train':
            np.random.shuffle(all_img_data)

        for img_data in all_img_data:
            try:

                if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
                    continue

                # read in image, and optionally add augmentation

                if mode == 'train':
                    img_data_aug, x_img = augment(img_data, C, augment=True)
                else:
                    img_data_aug, x_img = augment(img_data, C, augment=False)

                (width, height) = (img_data_aug['width'], img_data_aug['height'])
                (rows, cols, _) = x_img.shape

                assert cols == width
                assert rows == height

                # get image dimensions for resizing
                (resized_width, resized_height) = get_new_img_size(width, height, C.im_size)

                # resize the image so that smalles side is length = 600px
                x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

                try:
                    y_rpn_cls, y_rpn_regr = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height,
                                                     img_length_calc_function)
                except:
                    continue

                # Zero-center by mean pixel, and preprocess image

                x_img = x_img[:, :, (2, 1, 0)]  # BGR -> RGB
                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]
                x_img /= C.img_scaling_factor

                x_img = np.transpose(x_img, (2, 0, 1))
                x_img = np.expand_dims(x_img, axis=0)

                y_rpn_regr[:, y_rpn_regr.shape[1] // 2:, :, :] *= C.std_scaling

                if backend == 'tf':
                    x_img = np.transpose(x_img, (0, 2, 3, 1))
                    y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
                    y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

                yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug

            except Exception as e:
                print(e)
                continue


def augment(img_data, config, augment=True):
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    assert 'width' in img_data
    assert 'height' in img_data

    img_data_aug = copy.deepcopy(img_data)

    img = cv2.imread(img_data_aug['filepath'])
    img = cv2.resize(img, (GeneralFlags.train_image_width.value, GeneralFlags.train_image_height.value))
    if augment:
        rows, cols = img.shape[:2]

        if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 1)
            for bbox in img_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                bbox['x2'] = cols - x1
                bbox['x1'] = cols - x2

        if config.use_vertical_flips and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 0)
            for bbox in img_data_aug['bboxes']:
                y1 = bbox['y1']
                y2 = bbox['y2']
                bbox['y2'] = rows - y1
                bbox['y1'] = rows - y2

        if config.rot_90:
            angle = np.random.choice([0,90,180,270],1)[0]
            if angle == 270:
                img = np.transpose(img, (1,0,2))
                img = cv2.flip(img, 0)
            elif angle == 180:
                img = cv2.flip(img, -1)
            elif angle == 90:
                img = np.transpose(img, (1,0,2))
                img = cv2.flip(img, 1)
            elif angle == 0:
                pass

            for bbox in img_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                y1 = bbox['y1']
                y2 = bbox['y2']
                if angle == 270:
                    bbox['x1'] = y1
                    bbox['x2'] = y2
                    bbox['y1'] = cols - x2
                    bbox['y2'] = cols - x1
                elif angle == 180:
                    bbox['x2'] = cols - x1
                    bbox['x1'] = cols - x2
                    bbox['y2'] = rows - y1
                    bbox['y1'] = rows - y2
                elif angle == 90:
                    bbox['x1'] = rows - y2
                    bbox['x2'] = rows - y1
                    bbox['y1'] = x1
                    bbox['y2'] = x2
                elif angle == 0:
                    pass

    img_data_aug['width'] = img.shape[1]
    img_data_aug['height'] = img.shape[0]
    return img_data_aug, img


class DataSinif:
    id = -1
    sinifname = ""
    foldername = ""
    fiyat = 0
    doviz = ""
    pass


class SimpleModelFlags(Enum):
    buffer_size = 1000
    batch_size = 20
    init_filter = (3, 3)
    stride = (1, 1)
    save_path = None
    mode = 'from_depth'
    depth_of_model = 7
    growth_rate = 12
    num_of_blocks = 3
    output_classes = 10
    num_layers_in_each_block = -1
    data_format = 'channels_last'
    bottleneck = True
    compression = 0.5
    weight_decay = 1e-4
    dropout_rate = 0.
    pool_initial = False
    include_top = True
    train_mode = 'custom_loop'
    image_height = 120
    image_width = 120
    image_deep = 3


class DenseNetFlags(Enum):
    buffer_size = 1000
    batch_size = 10
    init_filter = (3, 3)
    stride = (1, 1)
    save_path = None
    mode = 'from_depth'
    depth_of_model = 7
    growth_rate = 12
    num_of_blocks = 3
    output_classes = 10
    num_layers_in_each_block = -1
    data_format = 'channels_last'
    bottleneck = True
    compression = 0.5
    weight_decay = 1e-4
    dropout_rate = 0.
    pool_initial = False
    include_top = True
    train_mode = 'custom_loop'
    image_height = 120
    image_width = 120
    image_deep = 3


class GeneralFlags(Enum):
    epoch = 3
    enable_function = False,
    train_mode = 'custom_loop'
    train_image_height = 600
    train_image_width = 800


class FasterRCNNConfig:
    def __init__(self):
        self.class_mapping = None
    verbose = True

    network = 'resnet50'

    # setting for data augmentation
    use_horizontal_flips = False
    use_vertical_flips = False
    rot_90 = False

    # anchor box scales
    anchor_box_scales = [128, 256, 512]

    # anchor box ratios
    anchor_box_ratios = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]

    # size to resize the smallest side of the image
    im_size = 600

    # image channel-wise mean to subtract
    img_channel_mean = [103.939, 116.779, 123.68]
    img_scaling_factor = 1.0

    # number of ROIs at once
    num_rois = 4

    # stride at the RPN (this depends on the network configuration)
    rpn_stride = 16

    balanced_classes = False

    # scaling the stdev
    std_scaling = 4.0
    classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

    # overlaps for RPN
    rpn_min_overlap = 0.3
    rpn_max_overlap = 0.7

    # overlaps for classifier ROIs
    classifier_min_overlap = 0.1
    classifier_max_overlap = 0.7

    def setClassMapping(self, value):
        self.class_mapping = value

    def getClassMapping(self):
        return self.class_mapping

    model_path = r'C:\Users\Durkan\Documents\GitHub\FoodRecognition\OutPuts\model_frcnn.resnet.hdf5'

    lambda_rpn_regr = 1.0
    lambda_rpn_class = 1.0

    lambda_cls_regr = 1.0
    lambda_cls_class = 1.0
    rpn_checkpoint_dir = r'C:\Users\Durkan\Documents\GitHub\FoodRecognition\OutPuts\RPNCheckpoints\rpn_checkpoints'
    classifier_checkpoint_dir = r'C:\Users\Durkan\Documents\GitHub\FoodRecognition\OutPuts\ClasifierCheckpoints\classifier_checkpoint'
    epsilon = 1e-4