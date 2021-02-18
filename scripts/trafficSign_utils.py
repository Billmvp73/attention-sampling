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
import json
import subprocess
from cv2 import imread, imwrite, VideoCapture, resize
import cv2
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

def draw_bbox(img, bboxs):
    for bbox in bboxs:
        img = cv2.rectangle(img, (int(bbox["x1"]), int(bbox["y1"])), (int(bbox["x2"]), int(bbox["y2"])), (0, 0, 255), 2)
    return img

def main(video_path, anno_path):
    video = VideoCapture(video_path)
    video_name = video_path[video_path.rfind('/')+1:-4]
    draw_path = video_path[:video_path.rfind("/")+1] + "draw_video_" + video_name
    if not os.path.exists(draw_path):
        os.mkdir(draw_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    count = 1
    anno_mapping_path = anno_path[:-5] + "_mapping.json"
    if not os.path.exists(anno_mapping_path):
        prefix_mappings = {}
    else:
        with open(anno_mapping_path) as f_map:
            prefix_mappings = json.load(f_map)
    with open(anno_path) as f:
        annotations = json.load(f)
        if not os.path.exists(anno_mapping_path):
            for anno in annotations:
                name = anno['name']
                prefix, img_name = name.split('/')
                pre_v = img_name.split('-')[0]
                if pre_v not in prefix_mappings:
                    prefix_mappings[pre_v] = prefix
                #print(prefix+" "+pre_v)
    if not os.path.exists(anno_mapping_path):
        with open(anno_mapping_path, "w") as mapping_file:
            json.dump(prefix_mappings, mapping_file)
    while(video.isOpened()):
        ret, frame = video.read()
        if frame is None:
            break
        frame_name = video_name + "-{0:07d}.jpg".format(count)
        if video_name in prefix_mappings:
            prefix = prefix_mappings[video_name]
            anno_name = prefix + '/' + frame_name
            for anno in annotations:
                if anno_name == anno["name"]:
                    bboxs = []
                    if anno["labels"] is None:
                        break
                    for item in anno["labels"]:
                        if "box2d" in item:
                            bboxs.append(item["box2d"])
                    draw_img = draw_bbox(frame, bboxs)
                    cv2.imwrite(draw_path +'/'+ frame_name, draw_img)
                    break
        count += 1
    print(count)
    video.release()

def printLabels(anno_path, num):
    with open(anno_path, 'r') as f:
        annotations = json.load(f)
    all_categories = []
    all_attributes = []
    all_labels = []
    for anno in annotations:
        if anno["labels"] == None:
            continue
        else:
            anno_label = anno["labels"]
            for label in anno_label:
                if label["category"] not in all_categories or label["attributes"] not in all_attributes:
                    #print(label["category"])
                    all_categories.append(label["category"])
                    #print(label["attributes"])
                    all_attributes.append(label["attributes"])
                    all_labels.append({
                        "category": label["category"],
                        "attributes": label["attributes"]
                    })
    # print("all categroies: ", all_categories)
    # print("all attributes: ", all_attributes)
    print("%d labels %d: "%(int(num), len(all_labels)))
    return all_labels

class Sign(namedtuple("Sign", ["visibility", "bbox", "name"])):
    """A sign object. Useful for making ground truth images as well as making
    the dataset."""
    @property
    def x_min(self):
        return self.bbox['x1']

    @property
    def x_max(self):
        return self.bbox['x2']

    @property
    def y_min(self):
        return self.bbox['y1']

    @property
    def y_max(self):
        return self.bbox['y2']

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
        visibilities = ["Illegible", "English_Legible", "Non_English_Legible"]
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

class RoadText1k:
    """ The RoadText1k class reads the annotations and creates the corresponding objects."""
    #Sets = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    sets = dict(
        val = [500, 600],
        train = [0, 100, 200, 300, 400],
        test = [700, 800, 900],
    )
    def __init__(self, directory, split="train", seed=0):
        self._directory = directory
        self._set = self.sets[split][seed%len(self.sets[split])]
        self._split  = split
        self._data = self._load_signs(self._directory +"/"+"Ground_truths/Localisation", self._set)

    def _load_signs(self, directory, inner):
        with open(os.path.join(directory, "{}_videos_results.json".format(inner))) as f:
            annotations = json.load(f)
        all_signs = []
        images = []
        for anno in annotations:
            video_num = anno["name"].split("/")[1].split('-')[0]
            num = os.path.join(self._directory + "/Videos/"+ self._split+'/'+ str(inner)+"/video_{}".format(video_num), anno["name"].split("/")[1])
            images.append(num)
            labels = anno["labels"]
            signs = []
            if labels is not None:
                for label in labels:
                    signs.append(Sign(label["category"], label["box2d"], label["attributes"]))
            all_signs.append(signs)
        return list(zip(images, all_signs))
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, i):
        return self._data[i]
    
class trafficSignDataSet(Sequence):
    """Provide a Keras Sequence for the RoadText-1K dataset.

    Arguments
    ---------
        directory: str, The directory that the dataset already is or is going
                   to be downloaded in
        train: bool, Select the training or testing sets
        seed: int, The prng seed for the dataset
    """
    CATERORIES = ["English_Text", "Licence_Plates"]

    CLASSES = ["EMPTY", *CATERORIES]

    def __init__(self, directory, split="train", seed=0):
        self._data = self._filter(RoadText1k(directory, split, seed))

    def _filter(self, data):
        filtered = []
        for image, signs in data:
            signs, acceptable = self._acceptable(signs)
            if acceptable and os.path.exists(image):
            #if os.path.exists(image):
                if not signs:
                    filtered.append((image, 0))
                else:
                    filtered.append((image, self.CLASSES.index(signs[0].name["English_Legible"][1])))
        return filtered
    
    def _acceptable(self, signs):
        if not signs:
            return signs, True
        
        signs = sorted([s for s in signs if s.name["English_Legible"][1] in self.CATERORIES and s.visibility != "Illegible"])

         # No speed limit but many other signs
        if not signs:
            return None, False

        # Not legible sign so skip
        if signs[0].visibility == "Illegible":
            return None, False

        return signs, True

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        image, category = self._data[i]
        data = imread(image)
        data = resize(data, (1280, 960))
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

def video2frame(dataset):
    splits = ["train", "test", "val"]
    for split in splits:
        split_path = os.path.join(dataset + "/Videos", split)
        num_list = os.listdir(split_path)
        for num in sorted(num_list):
            dir_path = os.path.join(split_path, num)
            video_list = os.listdir(dir_path)
            for video in sorted(video_list):
                if not video.endswith(".mp4"):
                    continue
                v_num = video.split('.')[0]
                new_path = os.path.join(dir_path, "video_" + v_num)
                if not os.path.exists(new_path):
                    os.mkdir(new_path)
                    frame = v_num + "-%07d.jpg"
                    video_capture = VideoCapture(os.path.join(dir_path, video))
                    fps = video_capture.get(cv2.CAP_PROP_FPS)
                    ffmpeg_args = [
                        "ffmpeg",
                        "-i", 
                        os.path.join(dir_path , video),
                        "-r",
                        str(fps),
                        os.path.join(new_path, frame)
                    ]
                    popenObj = subprocess.Popen(
                        args = ffmpeg_args
                    )
                    popenObj.wait()
                    os.remove(os.path.join(dir_path , video))

        #data = imread(image)


if __name__ == "__main__":
    #main("/media/pyhuang/E/RoadText1k/Videos/train/0/2.mp4", '/media/pyhuang/E/RoadText1k/Ground_truths/Localisation/0_videos_results.json')
    # anno_path = '/media/pyhuang/E/RoadText1k/Ground_truths/Localisation'
    # labels = []
    # for anno in os.listdir(anno_path):
    #     anno_labels = printLabels(path.join(anno_path, anno), anno.split('_')[0])
    #     for label in anno_labels:
    #         if label not in labels:
    #             labels.append(label)
    # print(labels)
    # print(len(labels))
    dataset = "/home/pyhuang/UM/saliency_2021/traffic_sign/RoadText1k"
    # train_set = trafficSignDataSet(dataset, split="train")
    # test_set = trafficSignDataSet(dataset, split="test")
    # training_batched = Batcher(train_set, 32)
    # test_batched = Batcher(test_set, 32)
    video2frame(dataset)
    