from collections import namedtuple
from functools import partial
import hashlib
import urllib.request
import os
from os import path
import string
import sys
import zipfile

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

from ats.core import attention_sampling
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
        if epoch < decrease_lr_at:
            return lr
        else:
            return lr * 0.1

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
    def __init__(self, directory, CLASSES, sets=None):

        self._directory = directory
        self._sets = sets
        self.CLASSES = CLASSES
        self.CLASSES_LEN = len(self.CLASSES)
        self.CLASS_TO_IDX = dict(zip(self.CLASSES, range(self.CLASSES_LEN)))
        self._data = self._load_files(self._directory, self._sets)

    def _load_files(self, directory, inner):
        img_path = os.path.join(directory, "image_2")
        label_path = os.path.join(directory, "label_2")
        images = []
        annotations = []
        if inner is None:
            inner = []
            for f in os.listdir(img_path):
                inner.append(f.split(".")[0])
            # inner = [f.split(".")[0] if f.endswith(".png") for f in os.listdir(img_path)]
        for name in inner:
            img = os.path.join(img_path, name + ".png")
            gt_file = os.path.join(label_path, name + ".txt")
            annotation = self._load_annotation(gt_file)
            if len(annotation) > 0:
                images.append(img)
                annotations.append(annotation)
        return list(zip(images, annotations))

    def _load_annotation(self, gt_file):
        with open(gt_file, 'r') as f:
            lines = f.readlines()
        f.close()

        annotations = []

        #each line is an annotation bounding box
        for line in lines:
            obj = line.strip().split(' ')

            #get class, if class is not in listed, skip it
            try:
                cls = self.CLASS_TO_IDX[obj[0].lower().strip()]
                # print cls

                #get coordinates
                xmin = float(obj[4])
                ymin = float(obj[5])
                xmax = float(obj[6])
                ymax = float(obj[7])


                #check for valid bounding boxes
                assert xmin >= 0.0 and xmin <= xmax, \
                    'Invalid bounding box x-coord xmin {} or xmax {} at {}' \
                        .format(xmin, xmax, gt_file)
                assert ymin >= 0.0 and ymin <= ymax, \
                    'Invalid bounding box y-coord ymin {} or ymax {} at {}' \
                        .format(ymin, ymax, gt_file)


                #transform to  point + width and height representation
                x, y, w, h = bbox_transform_inv(xmin, ymin, xmax, ymax)

                annotations.append([x, y, w, h, cls])

            except:
                continue
        if len(annotations) == 0:
            annotations.append([-1, -1, -1, -1, 0])
        return annotations


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

class KittiData(Sequence):
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
    CLASSES = ['empty', 'car', 'pedestrian', 'cyclist']

    def __init__(self, directory, split='train', sets=None):
        self.directory = directory
        if split == 'train' or split == 'val':
            self.directory = os.path.join(self.directory, "training")
        else:
            self.directory = os.path.join(self.directory, "testing")
        self.split = split
        self._data = self._filter(STS(self.directory, self.CLASSES, sets))


    def _filter(self, data):
        filtered = []
        for image, annos in data:
            # signs, acceptable = self._acceptable(signs)
            # if acceptable:
            #     if not signs:
            #         filtered.append((image, 0))
            #     else:
            #         filtered.append((image, self.CLASSES.index(signs[0].name)))
            categories = []
            for a in annos:
                if a[-1] not in categories:
                    categories.append(a[-1])
            # categories = [a[-1] for a in annos]
            filtered.append((image, categories))
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
        data = resize(data, (1242, 375))
        data = data.astype(np.float32) / np.float32(255.)
        # label = np.eye(len(self.CLASSES), dtype=np.float32)[category]
        label = np.zeros(len(self.CLASSES), dtype=np.float32)
        for c in category:
            label[c] = 1
        return data, label

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