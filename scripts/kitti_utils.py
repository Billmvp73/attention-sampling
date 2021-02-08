import argparse
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


def check_file(filepath, md5sum):
    """Check a file against an md5 hash value.

    Returns
    -------
        True if the file exists and has the given md5 sum False otherwise
    """
    try:
        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(partial(f.read, 4096), b""):
                md5.update(chunk)
        return md5.hexdigest() == md5sum
    except FileNotFoundError:
        return False


def download_file(url, destination, progress_file=sys.stderr):
    """Download a file with progress."""
    response = urllib.request.urlopen(url)
    n_bytes = response.headers.get("Content-Length")
    if n_bytes == "":
        n_bytes = 0
    else:
        n_bytes = int(n_bytes)

    message = "\rReceived {} / {}"
    cnt = 0
    with open(destination, "wb") as dst:
        while True:
            print(message.format(cnt, n_bytes), file=progress_file,
                  end="", flush=True)
            data = response.read(65535)
            if len(data) == 0:
                break
            dst.write(data)
            cnt += len(data)
    print(file=progress_file)


def ensure_dataset_exists(directory, tries=1, progress_file=sys.stderr):
    """Ensure that the dataset is downloaded and is correct.

    Correctness is checked only against the annotations files.
    """
    set1_url = ("http://www.isy.liu.se/cvl/research/trafficSigns"
                "/swedishSignsSummer/Set1/Set1Part0.zip")
    set1_annotations_url = ("http://www.isy.liu.se/cvl/research/trafficSigns"
                            "/swedishSignsSummer/Set1/annotations.txt")
    set1_annotations_md5 = "9106a905a86209c95dc9b51d12f520d6"
    set2_url = ("http://www.isy.liu.se/cvl/research/trafficSigns"
                "/swedishSignsSummer/Set2/Set2Part0.zip")
    set2_annotations_url = ("http://www.isy.liu.se/cvl/research/trafficSigns"
                            "/swedishSignsSummer/Set2/annotations.txt")
    set2_annotations_md5 = "09debbc67f6cd89c1e2a2688ad1d03ca"

    integrity = (
        check_file(
            path.join(directory, "Set1", "annotations.txt"),
            set1_annotations_md5
        ) and check_file(
            path.join(directory, "Set2", "annotations.txt"),
            set2_annotations_md5
        )
    )

    if integrity:
        return

    if tries <= 0:
        raise RuntimeError(("Cannot download dataset or dataset download "
                            "is corrupted"))

    print("Downloading Set1", file=progress_file)
    download_file(set1_url, path.join(directory, "Set1.zip"),
                  progress_file=progress_file)
    print("Extracting...", file=progress_file)
    with zipfile.ZipFile(path.join(directory, "Set1.zip")) as archive:
        archive.extractall(path.join(directory, "Set1"))
    print("Getting annotation file", file=progress_file)
    download_file(
        set1_annotations_url,
        path.join(directory, "Set1", "annotations.txt"),
        progress_file=progress_file
    )
    print("Downloading Set2", file=progress_file)
    download_file(set2_url, path.join(directory, "Set2.zip"),
                  progress_file=progress_file)
    print("Extracting...", file=progress_file)
    with zipfile.ZipFile(path.join(directory, "Set2.zip")) as archive:
        archive.extractall(path.join(directory, "Set2"))
    print("Getting annotation file", file=progress_file)
    download_file(
        set2_annotations_url,
        path.join(directory, "Set2", "annotations.txt"),
        progress_file=progress_file
    )

    return ensure_dataset_exists(
        directory,
        tries=tries-1,
        progress_file=progress_file
    )


class Sign(namedtuple("Sign", ["visibility", "bbox", "type", "name"])):
    """A sign object. Useful for making ground truth images as well as making
    the dataset."""
    @property
    def x_min(self):
        return self.bbox[2]

    @property
    def x_max(self):
        return self.bbox[0]

    @property
    def y_min(self):
        return self.bbox[3]

    @property
    def y_max(self):
        return self.bbox[1]

    @property
    def area(self):
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    @property
    def center(self):
        return [
            (self.y_max - self.y_min)/2 + self.y_min,
            (self.x_max - self.x_min)/2 + self.x_min
        ]

    @property
    def visibility_index(self):
        visibilities = ["VISIBLE", "BLURRED", "SIDE_ROAD", "OCCLUDED"]
        return visibilities.index(self.visibility)

    def pixels(self, scale, size):
        return zip(*(
            (i, j)
            for i in range(round(self.y_min*scale), round(self.y_max*scale)+1)
            for j in range(round(self.x_min*scale), round(self.x_max*scale)+1)
            if i < round(size[0]*scale) and j < round(size[1]*scale)
        ))

    def __lt__(self, other):
        if not isinstance(other, Sign):
            raise ValueError("Signs can only be compared to signs")

        if self.visibility_index != other.visibility_index:
            return self.visibility_index < other.visibility_index

        return self.area > other.area


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
                # x, y, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])
                x, y, w, h = (xmin, ymin, xmax, ymax)

                annotations.append([x, y, w, h, cls])

            except:
                continue
        return annotations

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]

    


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
    CLASSES = ['car', 'van', 'truck',
                     'pedestrian', 'cyclist' , 'dontcare']

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
            categories = [a[-1] for a in annos]
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

    def _load_annotation(gt_file):
        with open(gt_file, 'r') as f:
            lines = f.readlines()
        f.close()

        annotations = []

        #each line is an annotation bounding box
        for line in lines:
            obj = line.strip().split(' ')

            #get class, if class is not in listed, skip it
            try:
                cls = config.CLASS_TO_IDX[obj[0].lower().strip()]
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
                x, y, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])

                annotations.append([x, y, w, h, cls])

            except:
                continue
        return annotations

    @property
    def image_size(self):
        return self[0][0].shape[:2]

    @property
    def class_frequencies(self):
        """Compute and return the class specific frequencies."""
        freqs = np.zeros(len(self.CLASSES), dtype=np.float32)
        for image, category in self._data:
            freqs[category] += 1
        return freqs/len(self._data)

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