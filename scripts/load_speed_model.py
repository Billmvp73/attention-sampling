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
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras.utils import Sequence, plot_model
import numpy as np

from ats.core import attention_sampling
from ats.core import sample, SamplePatches
from ats.utils.layers import L2Normalize, ResizeImages, SampleSoftmax, \
    ImageLinearTransform, ImagePan, ActivityRegularizer
from ats.utils.regularizers import multinomial_entropy
from ats.utils.training import Batcher

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from tensorflow import ConfigProto
from tensorflow import InteractiveSession
from tensorflow.python import debug as tf_debug
from keras.callbacks import TensorBoard
import tensorflow as tf
from speed_utils import STS, Sign, SpeedLimits

import matplotlib.pyplot as plt
from timer import Timer

config = ConfigProto(device_count = {'GPU': 0})
# config = ConfigProto()
# config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
sess = K.get_session()
sess = tf_debug.LocalCLIDebugWrapperSession(sess)
K.set_session(sess)

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
        type=float,
        default=250,
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

    args = parser.parse_args(argv)

    # Load the data
    training_set = SpeedLimits(args.dataset, train=True)
    test_set = SpeedLimits(args.dataset, train=False)
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
    
    if args.resume:
        load_path = os.path.join(args.load_dir, "weights."+ str(args.load_epoch) + ".h5")
        model.load_weights(load_path)
        model.trainable = False
        model.compile(
        loss="categorical_crossentropy",
        optimizer=get_optimizer(args),
        metrics=["accuracy", "categorical_crossentropy"]
        )
    else:
        model.compile(
        loss="categorical_crossentropy",
        optimizer=get_optimizer(args),
        metrics=["accuracy", "categorical_crossentropy"]
        )
        plot_model(model, to_file=path.join(args.output, "model.png"))

        callbacks = [
            AttentionSaver(args.output, att_model, training_set),
            ModelCheckpoint(
                path.join(args.output, "weights.{epoch:02d}.h5"),
                save_weights_only=True
            ),
            LearningRateScheduler(get_lr_schedule(args)),
            TensorBoard(log_dir=args.output+"/logs", histogram_freq=0, batch_size=args.batch_size, write_grads=True, write_images=True)
        ]
        model.fit_generator(
            training_batched,
            validation_data=test_batched,
            epochs=args.epochs,
            class_weight=class_weights,
            callbacks=callbacks
        )
    # while len(model.layers) > 12:
    #     model.layers.pop()
    # model.summary()
    # model.compile(
    #     loss="categorical_crossentropy",
    #     optimizer=get_optimizer(args),
    #     metrics=["accuracy", "categorical_crossentropy"]
    # )
    #x_high = model.layers[2].output
    #x_low = model.layers[3].output
    #attention_map = model.layers[10].output
    intermediate_low = model.layers[3].output
    intermediate_sample = model.layers[11].output
    intermediate_attention_map = model.layers[10].output
    #model_b = Model(model.input, [x_low, x_high, attention_map])
    model_b = Model(model.input, [intermediate_low, intermediate_sample[0], intermediate_sample[1], intermediate_sample[2], intermediate_attention_map])
    if not os.path.exists("test_patch"):
        os.mkdir("test_patch")
    for inputs, targets in test_batched:
        #with tf.GradientTape() as tape:
        #Forward pass.
        timer_batch = Timer(desc="time per batch")
        timer_batch.start_time()
        low, patches, sampled_attention, samples, attention_map = model_b.predict(inputs)
        #model.evaluate()
        #loss = model.test_on_batch(inputs, targets)
        # low, high, ats_map = model_b.predict(inputs)
        # sample_space = K.shape(ats_map)[1:]
        # samples, sampled_attention = sample(args.n_patches, ats_map, sample_space, receptive_field=9, replace=False)
        timer_batch.end_time()
        # for b in range(patches.shape[0]):
        #     imgs = patches[b]
        #     sample = samples[b]
        #     fig, axs = plt.subplots(2)
        #     axs[0].imshow(low[b])
        #     axs[0].axis('off')
        #     axs[1].imshow(attention_map[b])
        #     axs[1].axis('off')
        #     plt.savefig("test_patch/input%d.png" % b)
        #     plt.clf()
        #     for i in range(imgs.shape[0]):
        #         patch = imgs[i]
        #         sample_i = sample[i]
        #         print("img %d sample %d location: "%(b, i), sample_i)
        #         plt.imshow(patch)
        #         plt.axis('off')
        #         plt.savefig("test_patch/b%d_patch_%d.png" % (b, i))
        #         plt.clf()
        break    
        # attention_regularizer = multinomial_entropy(args.regularizer_strength)
        # attention_map = ActivityRegularizer(attention_regularizer)(attention_map)




if __name__ == "__main__":
    main(None)

# new_model = load_model('/home/pyhuang/UM/attention/attention_sampling_keras/weights.1000.h5')
# new_model.summary()
# new_model.trainable = False    
# while len(new_model.layers) > 12:
#     new_model.layers.pop()
# test_img = 
