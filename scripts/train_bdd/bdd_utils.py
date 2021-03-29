from collections import namedtuple
from functools import partial
import hashlib
import urllib.request
import os
from os import path
import string
import sys
import zipfile
import json

from cv2 import imread, imwrite, resize
import keras.backend as K
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
from keras.layers import Activation, BatchNormalization, Conv2D, \
    GlobalAveragePooling2D, MaxPooling2D, Dense, Input, add
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras.utils import Sequence, plot_model
import numpy as np

from ats.core import attention_sampling, multi_attention_sampling
from ats.utils.layers import L2Normalize, ResizeImages, SampleSoftmax, \
    ImageLinearTransform, ImagePan
from ats.utils.regularizers import multinomial_entropy
from ats.utils.training import Batcher

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from tensorflow import ConfigProto
from tensorflow import InteractiveSession
from tensorflow.python import debug as tf_debug
from keras.callbacks import TensorBoard

class AttentionSaver(Callback):
    """Save the attention maps to monitor model evolution."""
    def __init__(self, output_directory, att_model, training_set, period=1):
        self._dir = path.join(output_directory, "attention")
        try:
            os.mkdir(self._dir)
        except FileExistsError:
            pass
        self._att_model = att_model
        idxs = training_set.strided(10)
        data = [training_set[i] for i in idxs]
        self._X = np.array([d[0] for d in data])
        self._Y = np.array([d[1] for d in data]).argmax(axis=1)
        np.savetxt(
            path.join(self._dir, "points.txt"),
            np.array([[i, yi] for i, yi in zip(idxs, self._Y)]).astype(int),
            fmt="%d"
        )
        self.period = period
        self.epoch_since_last_save = 0

    def on_train_begin(self, *args):
        _, _, x_low = self._att_model.predict(self._X)
        for i, xi in enumerate(x_low):
            self._imsave(path.join(self._dir, "{}.jpg").format(i), xi)

    def on_epoch_end(self, e, logs):
        self.epoch_since_last_save += 1
        if self.epoch_since_last_save >= self.period:
            self.epoch_since_last_save = 0
            att, patches, _ = self._att_model.predict(self._X)
            for i, att_i in enumerate(att):
                np.save(path.join(self._dir, "att_{}_{}.npy").format(e, i), att_i)

    def _imsave(self, filepath, x):
        x = (x*255).astype(np.uint8)
        imwrite(filepath, x)


def resnet(x, strides=[1, 2, 2, 2], filters=[32, 32, 32, 32]):
    """Implement a simple resnet."""
    # Do a convolution on x
    def c(x, filters, kernel, strides):
        return Conv2D(filters, kernel_size=kernel, strides=strides,
                      padding="same", use_bias=False)(x)

    # Do a BatchNorm on x
    def b(x):
        return BatchNormalization()(x)

    # Obviosuly just do relu
    def relu(x):
        return Activation("relu")(x)

    # Implement a resnet block. short is True when we need to add a convolution
    # for the shortcut
    def block(x, filters, strides, short):
        x = b(x)
        x = relu(x)
        x_short = x
        if short:
            x_short = c(x, filters, 1, strides)
        x = c(x, filters, 3, strides)
        x = b(x)
        x = relu(x)
        x = c(x, filters, 3, 1)
        x = add([x, x_short])

        return x

    # Implement the resnet
    stride_prev = strides.pop(0)
    filters_prev = filters.pop(0)
    y = c(x, filters_prev, 3, stride_prev)
    for s, f in zip(strides, filters):
        y = block(y, f, s, s != 1 or f != filters_prev)
        stride_prev = s
        filters_prev = f
    y = b(y)
    y = relu(y)

    # Average the final features and normalize them
    y = GlobalAveragePooling2D()(y)
    y = L2Normalize()(y)

    return y


def attention(x):
    params = dict(
        activation="relu",
        padding="valid",
        kernel_regularizer=l2(1e-5)
    )
    x = Conv2D(8, kernel_size=3, **params)(x)
    x = Conv2D(16, kernel_size=3, **params)(x)
    x = Conv2D(32, kernel_size=3, **params)(x)
    x = Conv2D(1, kernel_size=3)(x)
    x = MaxPooling2D(pool_size=8)(x)
    x = SampleSoftmax(squeeze_channels=True, smooth=1e-4)(x)

    return x


def get_model(outputs, width, height, scale, n_patches, patch_size, reg):
    x_in = Input(shape=(height, width, 3))
    x_high = ImageLinearTransform()(x_in)
    x_high = ImagePan(horizontally=True, vertically=True)(x_high)
    x_low = ResizeImages((int(height*scale), int(width*scale)))(x_high)

    features, att, patches = attention_sampling(
        attention,
        resnet,
        patch_size,
        n_patches,
        replace=False,
        attention_regularizer=multinomial_entropy(reg),
        receptive_field=9
    )([x_low, x_high])
    y = Dense(outputs, activation="softmax")(features)

    return (
        Model(inputs=x_in, outputs=[y]),
        Model(inputs=x_in, outputs=[att, patches, x_low])
    )

def get_multi_model(outputs, scales, width, height, scale, n_patches, patch_size, reg):
    x_in = Input(shape=(height, width, 3))
    
    ats_input_layers = []
    for res_scale in scales:
        x_high = ImageLinearTransform()(x_in)
        x_high = ImagePan(horizontally=True, vertically=True)(x_high)
        x_high = ResizeImages((int(height/res_scale), int(width/res_scale)))(x_high)
        x_low = ResizeImages((int(height/res_scale*scale), int(width/res_scale*scale)))(x_high)
        ats_input_layers.append([res_scale, x_low, x_high])


    features, att, patches = multi_attention_sampling(
        attention,
        resnet,
        patch_size,
        n_patches,
        replace=False,
        attention_regularizer=multinomial_entropy(reg),
        receptive_field=9
    )(ats_input_layers)
    y = Dense(outputs, activation="softmax")(features)

    return (
        Model(inputs=x_in, outputs=[y]),
        Model(inputs=x_in, outputs=[att, patches, x_low])
    )

def get_optimizer(args):
    optimizer = args.optimizer

    if optimizer == "sgd":
        return SGD(lr=args.lr, momentum=args.momentum, clipnorm=args.clipnorm)
    elif optimizer == "adam":
        return Adam(lr=args.lr, clipnorm=args.clipnorm)

    raise ValueError("Invalid optimizer {}".format(optimizer))


def get_lr_schedule(args):
    lr = args.lr
    decrease_lr_at = args.decrease_lr_at

    def get_lr(epoch):
        # decrease_lr_at.append(args.epochs)
        scale = 1
        for i, d_epoch in enumerate(decrease_lr_at):
            if epoch > d_epoch:
                scale *= 0.3
            else:
                break
        return lr * scale
        # for d_epoch in decrease_lr_at
        # if epoch < decrease_lr_at:
        #     return lr
        # else:
        #     return lr * 0.1

    return get_lr


class CropObj(namedtuple("CropObj", ["bbox", "name"])):
    """A cropped object. Useful for making ground truth images as well as making
    the dataset."""
    @property
    def x_min(self):
        return int(round(self.bbox[0]))

    @property
    def x_max(self):
        return int(round(self.bbox[2]))

    @property
    def y_min(self):
        return int(round(self.bbox[1]))

    @property
    def y_max(self):
        return int(round(self.bbox[3]))

    @property
    def width(self):
        return self.x_max - self.x_min

    @property
    def height(self):
        return self.y_max - self.y_min

    @property
    def area(self):
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    @property
    def size(self):
        return [self.width, self.height]

    @property
    def center(self):
        return [
            self.x_min + self.width / 2, 
            self.y_min + self.height / 2
        ]

    def pixels(self, scale, size):
        return zip(*(
            (i, j)
            for i in range(round(self.y_min*scale), round(self.y_max*scale)+1)
            for j in range(round(self.x_min*scale), round(self.x_max*scale)+1)
            if i < round(size[0]*scale) and j < round(size[1]*scale)
        ))
    
    def contains(self, another):
        return (self.x_min <= another.x_min and self.y_min <= another.y_min
                and self.x_min + self.width >= another.x_min + another.width
                and self.y_min + self.height >= another.y_min + another.height)


class STS:
    """The STS class reads the annotations and creates the corresponding
    Sign objects."""
    def __init__(self, directory, CLASSES_TO_IDX, annotations=None):

        self._directory = directory
        # self.CLASSES = CLASSES
        # self.CLASSES_LEN = len(self.CLASSES)
        # self.CLASS_TO_IDX = dict(zip(self.CLASSES, range(self.CLASSES_LEN)))
        self.CLASSES_TO_IDX = CLASSES_TO_IDX
        self.annotations = annotations
        self._data = self._load_files()

    def _load_files(self):
        with open(self.annotations, "r") as f:
            annotations = json.load(f)
        images = []
        final_annotations = []
        for anno in annotations:
            name = anno["name"]
            img_path = os.path.join(self._directory, name)
            images.append(img_path)
            img_labels = anno["labels"]
            target_anno = []
            for img_label in img_labels:
                if img_label["category"] not in self.CLASSES_TO_IDX:
                    continue
                attributes = img_label["attributes"]
                if not attributes["occluded"] and not attributes["truncated"]:
                    box2d = img_label["box2d"]
                    target_anno.append([float(box2d["x1"]), float(box2d["y1"]), float(box2d["x2"]), float(box2d["y2"]), self.CLASSES_TO_IDX[img_label["category"]]])
            if len(target_anno) == 0:
                target_anno.append([-1, -1, -1, -1, 0])
            final_annotations.append(target_anno)
        return list(zip(images, final_annotations))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]

def bbox_transform_inv(xmin, ymin, xmax, ymax):
    x_min = int(round(xmin))
    width = int(round(xmax)) - x_min
    y_min = int(round(ymin))
    height = int(round(ymax) - y_min)
    return [x_min, y_min, width, height]

class bddData(Sequence):
    """Provide a Keras Sequence for the SpeedLimits dataset which is basically
    a filtered version of the STS dataset.

    Arguments
    ---------
        directory: str, The directory that the dataset already is or is going
                   to be downloaded in
        train: bool, Select the training or testing sets
        seed: int, The prng seed for the dataset
    """
    #LIMITS = ["50_SIGN", "70_SIGN", "80_SIGN"]
    # CLASSES = ['empty', "bike", "bus", "car", "motor", "person", "rider", "truck"]
    CLASSES_TO_IDX = {
        "empty": 0,
        "car": 1,
        "bus": 1,
        "truck": 1,
        "train": 1,
        "bike": 2,
        "person": 2,
        "rider": 2,
        "motor": 2,
        "traffic light": 3,
        "traffic sign": 3
    }
    CLASSES = [0, 1]

    def __init__(self, directory, split='train', objects=None):
        self.directory = directory
        # if split == 'train' or split == 'val':
        #     self.directory = os.path.join(self.directory, "training")
        # else:
        #     self.directory = os.path.join(self.directory, "testing")
        self.split = split
        self.anno_path = None
        self.objects = objects
        if self.split == "train" or self.split == "val":
            self.anno_path = os.path.join(self.directory, "bdd100k_labels_images_{}.json".format(split))
        self.directory = os.path.join(self.directory, split)
        self._data = self._filter(STS(self.directory, self.CLASSES_TO_IDX, self.anno_path), self.objects)


    def _filter(self, data, objects):
        filtered = []
        for image, annos in data:
            # signs, acceptable = self._acceptable(signs)
            # if acceptable:
            #     if not signs:
            #         filtered.append((image, 0))
            #     else:
            #         filtered.append((image, self.CLASSES.index(signs[0].name)))
            if objects is None:
                categories = []
                large_car = False
                for a in annos:
                    if a[-1] not in categories:
                        categories.append(a[-1])
                    if self.CLASSES_TO_IDX["car"] == a[-1] and (abs(a[0] - a[2]) > 300 or abs(a[1] - a[3])):
                        large_car = True
                        break
                    else:
                        large_car = False
                if self.CLASSES_TO_IDX["car"] not in categories or large_car:
                    categories = [0]
                if len(categories) == 1:
                # categories = [a[-1] for a in annos]
                    filtered.append((image, categories))
            else:
                filtered.append((image, annos))
        return filtered

    # def _acceptable(self, signs):
    #     # Keep it as empty
    #     if not signs:
    #         return signs, True

    #     # Filter just the speed limits and sort them wrt visibility
    #     signs = sorted(s for s in signs if s.name in self.LIMITS)

    #     # No speed limit but many other signs
    #     if not signs:
    #         return None, False

    #     # Not visible sign so skip
    #     if signs[0].visibility != "VISIBLE":
    #         return None, False

    #     return signs, True

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        image, category = self._data[i]
        data = imread(image)
        # data = resize(data, (1242, 375))
        data = data.astype(np.float32) / np.float32(255.)
        # label = np.eye(len(self.CLASSES), dtype=np.float32)[category]
        if self.objects is None:
            label = np.zeros(len(self.CLASSES), dtype=np.float32)
            for c in category:
                label[c] = 1
            return data, label
        else:
            return data, np.array(category)

    @property
    def image_size(self):
        return self[0][0].shape[:2]

    @property
    def class_frequencies(self):
        """Compute and return the class specific frequencies."""
        freqs = np.zeros(len(self.CLASSES), dtype=np.float32)
        all_sum = 0
        for image, category in self._data:
            # freqs[category] += 1
            all_sum += len(category)
            for c in category:
                freqs[c] += 1
        print("one-hot vector: ", freqs)
        if self.objects is not None:
            return freqs
        return freqs/all_sum
        # return freqs/len(self._data)

    def strided(self, N):
        """Extract N images almost in equal proportions from each category."""
        order = np.arange(len(self._data))
        np.random.shuffle(order)
        idxs = []
        cat = 0
        while len(idxs) < N:
            for i in order:
                image, category = self._data[i]
                if cat in category:
                    idxs.append(i)
                    cat = (cat + 1) % len(self.CLASSES)
                if len(idxs) >= N:
                    break
        return idxs