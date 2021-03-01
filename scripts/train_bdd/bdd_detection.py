#!/usr/bin/env python
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import argparse
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
from bdd_utils import *

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
sess = K.get_session()
sess = tf_debug.LocalCLIDebugWrapperSession(sess)
K.set_session(sess)

def main(argv):
    parser = argparse.ArgumentParser(
        description=("Fetch the Sweidish Traffic Signs dataset and parse "
                     "it into the Speed Limits dataset subset")
    )
    parser.add_argument(
        "dataset",
        help="The location to download the dataset to"
    )
    parser.add_argument(
        "output",
        help="An output directory"
    )

    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adam"],
        default="adam",
        help="Choose the optimizer for Q1"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Set the optimizer's learning rate"
    )
    parser.add_argument(
        "--clipnorm",
        type=float,
        default=1,
        help="Clip the norm of the gradient to avoid exploding gradients"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Choose the momentum for the optimizer"
    )
    parser.add_argument(
        "--decrease_lr_at",
        type=lambda x: [int(xi) for xi in x.split(',')],
        default="250,500,750",
        help="Decrease the learning rate in this epoch"
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=0.3,
        help="How much to downscale the image for computing the attention"
    )
    parser.add_argument(
        "--patch_size",
        type=lambda x: tuple(int(xi) for xi in x.split("x")),
        default="100x100",
        help="Choose the size of the patch to extract from the high resolution"
    )
    parser.add_argument(
        "--n_patches",
        type=int,
        default=5,
        help="How many patches to sample"
    )
    parser.add_argument(
        "--regularizer_strength",
        type=float,
        default=0.0001,
        help="How strong should the regularization be for the attention"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Choose the batch size for SGD"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="How many epochs to train for"
    )

    parser.add_argument(
        "--resume",
        type=bool,
        default=False,
        help="Load pre-trained model"
    )

    parser.add_argument(
        "--load_epoch",
        type=int,
        default=0,
        help="Load pre-trained model at epoch number"
    )

    parser.add_argument(
        "--load_dir",
        type=str,
        default="",
        help="the path to load pre-trained model"
    )

    parser.add_argument(
        "--split_percent",
        type=int,
        default=80,
        help="the percent to split training set"
    )

    args = parser.parse_args(argv)
    folder = ""
    for d_epoch in args.decrease_lr_at:
        folder += "_" + str(d_epoch)
    args.output = path.join(args.output, folder + "0.3_" + str(args.n_patches))
    print("Output Directory: %s" % args.output)
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    # Load the data
    # training_set = SpeedLimits(args.dataset, train=True)
    # test_set = SpeedLimits(args.dataset, train=False)
    # bdd_data = [s.split(".")[0] for s in sorted(os.listdir(os.path.join(args.dataset, "training/image_2")))]
    # split_num = round(len(bdd_data) * args.split_percent / 100)
    # train_split = bdd_data[:split_num]
    # test_split = bdd_data[split_num:]
    training_set = bddData(args.dataset, 'train')
    test_set = bddData(args.dataset, "val")

    training_batched = Batcher(training_set, args.batch_size)
    test_batched = Batcher(test_set, args.batch_size)

    # Create the models
    H, W = training_set.image_size
    class_weights = training_set.class_frequencies
    class_weights = (1./len(class_weights)) / class_weights
    model, att_model = get_model(
        len(class_weights),
        W, H,
        args.scale,
        args.n_patches,
        args.patch_size,
        args.regularizer_strength
    )
    print(model.summary())
    model.compile(
        loss="categorical_crossentropy",
        optimizer=get_optimizer(args),
        metrics=["accuracy", "categorical_crossentropy"]
    )
    if args.resume:
        load_path = os.path.join(args.load_dir, "weights."+ str(args.load_epoch) + ".h5")
        model.load_weights(load_path)


    plot_model(model, to_file=path.join(args.output, "model.png"))

    callbacks = [
        # AttentionSaver(args.output, att_model, training_set),
        ModelCheckpoint(
            path.join(args.output, "weights.{epoch:02d}.h5"),
            save_weights_only=True
        ),
        LearningRateScheduler(get_lr_schedule(args)),
        TensorBoard(log_dir=args.output+"/logs", histogram_freq=0, batch_size=args.batch_size, write_grads=True, write_images=True)
    ]
    if args.resume:
        model.fit_generator(
            training_batched,
            validation_data=test_batched,
            epochs=args.epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            initial_epoch = int(args.load_epoch)
        )
    else:
        model.fit_generator(
            training_batched,
            validation_data=test_batched,
            epochs=args.epochs,
            class_weight=class_weights,
            callbacks=callbacks
        )

if __name__ == "__main__":
    main(None)
