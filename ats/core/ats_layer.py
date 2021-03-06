#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""The simplest possible interface for ATS. A layer that can be used in any
Keras model."""

from keras import backend as K
from keras.engine import Layer
from keras.layers import Lambda
import tensorflow as tf

from ..data.from_tensors import FromTensors
from ..utils.layers import ActivityRegularizer, TotalReshape
from .builder import ATSBuilder
from .sampling import sample
from .expectation import expected


class SamplePatches(Layer):
    """SamplePatches samples from a high resolution image using an attention
    map.

    The layer expects the following inputs when called `x_low`, `x_high`,
    `attention`. `x_low` corresponds to the low resolution view of the image
    which is used to derive the mapping from low resolution to high. `x_high`
    is the tensor from which we extract patches. `attention` is an attention
    map that is computed from `x_low`.

    Arguments
    ---------
        n_patches: int, how many patches should be sampled
        replace: bool, whether we should sample with replacement or without
        receptive_field: int, how large is the receptive field of the attention
                         network. It is used to map the attention to high
                         resolution patches.
    """
    def __init__(self, n_patches, patch_size, receptive_field=0, replace=False,
                 use_logits=False, **kwargs):
        self._n_patches = n_patches
        self._patch_size = patch_size
        self._receptive_field = receptive_field
        self._replace = replace
        self._use_logits = use_logits

        super(SamplePatches, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        shape_low, shape_high, shape_att = input_shape

        # Figure out the shape of the patches
        if K.image_data_format() == "channels_first":
            patch_shape = (shape_high[1], *self._patch_size)
        else:
            patch_shape = (*self._patch_size, shape_high[-1])
        patches_shape = (shape_high[0], self._n_patches, *patch_shape)

        # Sampled attention
        att_shape = (shape_high[0], self._n_patches)

        # samples shape
        samples_shape = (shape_att[0], self._n_patches, shape_att[1:][0])
        # return [patches_shape, att_shape, samples_shape]
        return [patches_shape, att_shape, samples_shape, samples_shape, ]

    def call(self, x):
        x_low, x_high, attention = x

        sample_space = K.shape(attention)[1:]
        samples, sampled_attention, indices = sample(
            self._n_patches,
            attention,
            sample_space,
            replace=self._replace,
            use_logits=self._use_logits
        )

        offsets = K.zeros(K.shape(samples), dtype="float32")
        if self._receptive_field > 0:
            offsets = offsets + self._receptive_field/2

        # Get the patches from the high resolution data
        # Make sure that below works
        assert K.image_data_format() == "channels_last"
        patches, offsets = FromTensors([x_low, x_high], None).patches(
            samples,
            offsets,
            sample_space,
            K.shape(x_low)[1:-1] - self._receptive_field,
            self._patch_size,
            0,
            1
        )

        # return [patches, sampled_attention, samples]
        return [patches, sampled_attention, offsets, samples]

class MultiSamplePatches(Layer):
    """SamplePatches samples from a high resolution image using an attention
    map.

    The layer expects the following inputs when called `x_low`, `x_high`,
    `attention`. `x_low` corresponds to the low resolution view of the image
    which is used to derive the mapping from low resolution to high. `x_high`
    is the tensor from which we extract patches. `attention` is an attention
    map that is computed from `x_low`.

    Arguments
    ---------
        n_patches: int, how many patches should be sampled
        replace: bool, whether we should sample with replacement or without
        receptive_field: int, how large is the receptive field of the attention
                         network. It is used to map the attention to high
                         resolution patches.
    """
    def __init__(self, n_patches, patch_size, receptive_field=0, replace=False,
                 use_logits=False, **kwargs):
        self._n_patches = n_patches
        self._patch_size = patch_size
        self._receptive_field = receptive_field
        self._replace = replace
        self._use_logits = use_logits

        super(SamplePatches, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        shape_low, shape_high, shape_att = input_shape

        # Figure out the shape of the patches
        if K.image_data_format() == "channels_first":
            patch_shape = (shape_high[1], *self._patch_size)
        else:
            patch_shape = (*self._patch_size, shape_high[-1])
        patches_shape = (shape_high[0], self._n_patches, *patch_shape)

        # Sampled attention
        att_shape = (shape_high[0], self._n_patches)

        # samples shape
        samples_shape = (shape_att[0], self._n_patches, shape_att[1:][0])
        # return [patches_shape, att_shape, samples_shape]
        return [patches_shape, att_shape, samples_shape, samples_shape]

    def call(self, x):
        x_lows, x_highs, ats_map, ats_index, attentions, scales = x

        sample_space = K.shape(ats_map)[1:]
        samples, sampled_attention, indices = sample(
            self._n_patches,
            ats_map,
            sample_space,
            replace=self._replace,
            use_logits=self._use_logits
        )
        # samples shape: (batch_size, n_samples, n_dims)
        batch_size, n_samples, n_dims = K.shape(samples)
        offsets = K.zeros(K.shape(samples), dtype="float32")
        # if self._receptive_field > 0:
        #     offsets = offsets + self._receptive_field/2
        res_index = K.tf.gather_nd(ats_index, indices)

        patches = K.zeros((batch_size, n_samples, self._patch_size, self._patch_size, 3), dtype="float32")
        for i in range(len(scales)):
            scale = scales[i]
            x_low = x_lows[i]
            x_high = x_highs[i]
            wh = K.tf.where(K.tf.equal(res_index, i))
            sample_i = K.tf.gather_nd(samples, wh)
            sample_i[:, :, 0] /= scale
            sample_i[:, :, 1] /= scale
            sample_space_i = K.tf.gather_nd(sample_space, wh)
            offset_i = K.zeros(K.shape(sample_i), dtype="float32")
            if self._receptive_field > 0:
                offset_i = offset_i + self._receptive_field/2
            
            assert K.image_data_format() == "channels_last"
            patches_i, offset_i = FromTensors([x_low, x_high], None).patches(
                sample_i,
                offset_i,
                sample_space_i,
                K.shape(x_low)[1:-1] - self._receptive_field,
                self._patch_size,
                0,
                1
            )
            patches = K.tf.scatter_nd_update(patches, wh, patches_i)
            offsets = K.tf.scatter_nd_update(offsets, wh, offsets_i)
            samples = K.tf.scatter_nd_update(samples, wh, samples_i)

        # return [patches, sampled_attention, samples]
        return [patches, sampled_attention, offsets, samples]


class Expectation(Layer):
    """Expectation averages the features in a way that gradients can be
    computed for both the features and the attention. See "Processing Megapixel
    Images with Deep Attention-Sampling Models"
    (https://arxiv.org/abs/1905.03711).

    Arguments
    ---------
        replace: bool, whether we should sample with replacement or without
    """
    def __init__(self, replace=False, **kwargs):
        self._replace = replace
        super(Expectation, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        features_shape, attention_shape = input_shape
        return (features_shape[0], features_shape[2])

    def call(self, x):
        features, sampled_attention = x
        return expected(
            sampled_attention,
            features,
            replace=self._replace
        )


def attention_sampling(attention, feature, patch_size=None, n_patches=10,
                       replace=False, attention_regularizer=None,
                       receptive_field=0):
    """Use attention sampling to process a high resolution image in patches.

    This function is meant to be a convenient way to use the layers defined in
    this module with Keras models or callables.

    Arguments
    ---------
        attention: A Keras layer or callable that takes a low resolution tensor
                   and returns and attention tensor
        feature: A Keras layer or callable that takes patches and returns
                 features
        patch_size: Tuple or tensor defining the size of the patches to be
                    extracted. If not given we try to extract it from the input
                    shape of the feature layer.
        n_patches: int that defines how many patches to extract from each
                   sample
        replace: bool, whether we should sample with replacement or without
        attention_regularizer: A regularizer callable for the attention
                               distribution
        receptive_field: int, how large is the receptive field of the attention
                         network. It is used to map the attention to high
                         resolution patches.

    Returns
    -------
        In the spirit of Keras we return a function that expects two tensors
        and returns three, namely the `expected features`, `attention` and
        `patches`

        ([x_low, x_high]) -> [expected_features, attention, patches]
    """
    if receptive_field is None:
        raise NotImplementedError(("Receptive field inference is not "
                                   "implemented yet"))

    if patch_size is None:
        if not isinstance(feature, Layer):
            raise ValueError(("Cannot infer patch_size if the feature "
                              "function is not a Keras Layer"))
        patch_size = list(feature.get_input_shape_at(0)[1:])
        patch_size.pop(-1 if K.image_data_format() == "channels_last" else 0)
        if any(s is None for s in patch_size):
            raise ValueError("Inferred patch size contains None")

    def apply_ats(x):
        assert isinstance(x, list) and len(x) == 2
        x_low, x_high = x

        # First we compute our attention map
        attention_map = attention(x_low)
        if attention_regularizer is not None:
            attention_map = \
                ActivityRegularizer(attention_regularizer)(attention_map)

        # Then we sample patches based on the attention
        # patches, sampled_attention, samples = SamplePatches(
        patches, sampled_attention, high_samples, samples = SamplePatches(
            n_patches,
            patch_size,
            receptive_field,
            replace
        )([x_low, x_high, attention_map])

        # We compute the features of the sampled patches
        channels = K.int_shape(patches)[-1]
        patches_flat = TotalReshape((-1, *patch_size, channels))(patches)
        patch_features = feature(patches_flat)
        dims = K.int_shape(patch_features)[-1]
        patch_features = TotalReshape((-1, n_patches, dims))(patch_features)

        # Finally we compute the expected features
        sample_features = Expectation(replace)([
            patch_features,
            sampled_attention
        ])

        # for one point, a patch for high resolution (100, 100), a patch for down-4 resolution (100,100)
        # ats (the total sum is 1.)

        # two ats
        # a: H, L : -> aH, aL -> aH -> gH
        # b: 

        return [sample_features, attention_map, patches]

    return apply_ats




def multi_attention_sampling(attention, feature, patch_size=None, n_patches=10,
                       replace=False, attention_regularizer=None,
                       receptive_field=0):
    """Use attention sampling to process a high resolution image in patches.

    This function is meant to be a convenient way to use the layers defined in
    this module with Keras models or callables.

    Arguments
    ---------
        attention: A Keras layer or callable that takes a low resolution tensor
                   and returns and attention tensor
        feature: A Keras layer or callable that takes patches and returns
                 features
        patch_size: Tuple or tensor defining the size of the patches to be
                    extracted. If not given we try to extract it from the input
                    shape of the feature layer.
        n_patches: int that defines how many patches to extract from each
                   sample
        replace: bool, whether we should sample with replacement or without
        attention_regularizer: A regularizer callable for the attention
                               distribution
        receptive_field: int, how large is the receptive field of the attention
                         network. It is used to map the attention to high
                         resolution patches.

    Returns
    -------
        In the spirit of Keras we return a function that expects two tensors
        and returns three, namely the `expected features`, `attention` and
        `patches`

        ([[scale, x_low, x_high]]) -> [expected_features, attention, patches]
    """
    if receptive_field is None:
        raise NotImplementedError(("Receptive field inference is not "
                                   "implemented yet"))

    if patch_size is None:
        if not isinstance(feature, Layer):
            raise ValueError(("Cannot infer patch_size if the feature "
                              "function is not a Keras Layer"))
        patch_size = list(feature.get_input_shape_at(0)[1:])
        patch_size.pop(-1 if K.image_data_format() == "channels_last" else 0)
        if any(s is None for s in patch_size):
            raise ValueError("Inferred patch size contains None")

    def apply_ats(x):
        assert isinstance(x, list)
        scales = []
        ats_maps = []
        x_lows = []
        x_highs = []
        for set_x in x:
            assert len(set_x) == 3
            scale, x_low, x_high = set_x
            scales.append(scale)
            x_lows.append(x_low)
            x_highs.append(x_high)
            # First we compute our attention map
            attention_map = attention(x_low)
            if attention_regularizer is not None:
                attention_map = \
                    ActivityRegularizer(attention_regularizer)(attention_map)
            ats_maps.append(attention_map)
        final_map = K.zeros_like(ats_maps[0])
        final_idx = K.zeros_like(ats_maps[0])
        # final_map = K.expand_dims(final_map, axis=1)
        # final_idx = K.expand_dims(final_idx, axis=1)

        upsample_ats = []
        def ResizeLayerLike(tensorA, tensorB):
            sB = K.int_shape(tensorB)
            print("sB size: ", sB)
            print("tensorA size: ", K.int_shape(tensorA))
            def resize_like(tensor, sB): 
                return tf.image.resize(tensor, sB[1:3], method=ResizedMethod.BILINEAR)
            return Lambda(resize_like, arguments={'sB':sB})(tensorA)

        for i, scale in enumerate(scales):
            # ats_maps[i] = K.expand_dims(ats_maps[i], axis=1)
            upsample_ats_i = tf.keras.layers.UpSampling2D(size=(scale, scale))(ats_maps[i])
            upsample_ats.append(upsample_ats_i)
            print(K.int_shape(upsample_ats_i))
        ats_all = K.stack(upsample_ats, axis=0)
        print(ats_all.shape)
        # for b in range(map_shape[0]):
        #     for i in range(map_shape[1]):
        #         for j in range(map_shape[2]):
        #             weights_xy = []
        #             for p, ats_map in enumerate(ats_maps):
        #                 scale = scales[p]
        #                 weights_xy.append(ats_map[b, i/scale, j/scale])
        #             final_map[b, i, j] = max(weights_xy)
        #             final_idx[b, i, j] = weights_xy.index(max(weights_xy))
        
        final_map = L2Normalize()(final_map)
        # Then we sample patches based on the attention
        # patches, sampled_attention, samples = SamplePatches(
        patches, sampled_attention, high_samples, samples = MultiSamplePatches(
            n_patches,
            patch_size,
            receptive_field,
            replace
        )([x_lows, x_highs, final_map, final_idx, ats_map, scales])

        # We compute the features of the sampled patches
        channels = K.int_shape(patches)[-1]
        patches_flat = TotalReshape((-1, *patch_size, channels))(patches)
        patch_features = feature(patches_flat)
        dims = K.int_shape(patch_features)[-1]
        patch_features = TotalReshape((-1, n_patches, dims))(patch_features)

        # Finally we compute the expected features
        sample_features = Expectation(replace)([
            patch_features,
            sampled_attention
        ])

        return [sample_features, attention_map, patches]

    return apply_ats
