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

from cv2 import imread, imwrite
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
    def __init__(self, directory, train=True, seed=0):
        ensure_dataset_exists(directory)

        self._directory = directory
        self._inner = "Set{}".format(1 + ((seed + 1 + int(train)) % 2))
        self._data = self._load_signs(self._directory, self._inner)

    def _load_files(self, directory, inner):
        files = set()
        with open(path.join(directory, inner, "annotations.txt")) as f:
            for l in f:
                files.add(l.split(":", 1)[0])
        return sorted(files)

    def _read_bbox(self, parts):
        def _float(x):
            try:
                return float(x)
            except ValueError:
                if len(x) > 0:
                    return _float(x[:-1])
                raise
        return [_float(x) for x in parts]

    def _load_signs(self, directory, inner):
        with open(path.join(directory, inner, "annotations.txt")) as f:
            lines = [l.strip() for l in f]
        keys, values = zip(*(l.split(":", 1) for l in lines))
        all_signs = []
        for v in values:
            signs = []
            for sign in v.split(";"):
                if sign == [""] or sign == "":
                    continue
                parts = [s.strip() for s in sign.split(",")]
                if parts[0] == "MISC_SIGNS":
                    continue
                signs.append(Sign(
                    visibility=parts[0],
                    bbox=self._read_bbox(parts[1:5]),
                    type=parts[5],
                    name=parts[6]
                ))
            all_signs.append(signs)
        images = [path.join(directory, inner, f) for f in keys]

        return list(zip(images, all_signs))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]


class SpeedLimits(Sequence):
    """Provide a Keras Sequence for the SpeedLimits dataset which is basically
    a filtered version of the STS dataset.

    Arguments
    ---------
        directory: str, The directory that the dataset already is or is going
                   to be downloaded in
        train: bool, Select the training or testing sets
        seed: int, The prng seed for the dataset
    """
    LIMITS = ["50_SIGN", "70_SIGN", "80_SIGN"]
    CLASSES = ["EMPTY", *LIMITS]

    def __init__(self, directory, train=True, seed=0):
        self._data = self._filter(STS(directory, train, seed))

    def _filter(self, data):
        filtered = []
        for image, signs in data:
            signs, acceptable = self._acceptable(signs)
            if acceptable:
                if not signs:
                    filtered.append((image, 0))
                else:
                    filtered.append((image, self.CLASSES.index(signs[0].name)))
        return filtered

    def _acceptable(self, signs):
        # Keep it as empty
        if not signs:
            return signs, True

        # Filter just the speed limits and sort them wrt visibility
        signs = sorted(s for s in signs if s.name in self.LIMITS)

        # No speed limit but many other signs
        if not signs:
            return None, False

        # Not visible sign so skip
        if signs[0].visibility != "VISIBLE":
            return None, False

        return signs, True

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        image, category = self._data[i]
        data = imread(image)
        data = data.astype(np.float32) / np.float32(255.)
        label = np.eye(len(self.CLASSES), dtype=np.float32)[category]

        return data, label

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
                if cat == category:
                    idxs.append(i)
                    cat = (cat + 1) % len(self.CLASSES)
                if len(idxs) >= N:
                    break
        return idxs