"""
Custom tensorflow / keras functions & operations
"""
import math

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
import numpy as np


def interpolate(im, x=None, y=None, out_size=None, opName=None):
    """
    Interpolates the input tensor at the specified positions
    Args:
        im: 3D tensor to interpolate
        x: x coord of interpolation positions
        y: y coord of interpolation positions
        out_size:
        opName:

    Returns:
        output: interpolated values
    """
    # Credit due to daviddao - modified from tensorflow STN implementation
    num_batch = tf.shape(im)[0]
    height = tf.shape(im)[1]
    width = tf.shape(im)[2]
    channels = tf.shape(im)[3]

    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')

    height_f = tf.cast(height, 'float32')
    width_f = tf.cast(width, 'float32')
    zero = tf.zeros([], dtype='int32')
    max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
    max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

    # scale indices from [-1, 1] to [0, width/height]
    #x = (x + 1.0)*(width_f) / 2.0
    #y = (y + 1.0)*(height_f) / 2.0

    # do sampling
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    dim2 = width
    dim1 = width*height
    base = _repeat(tf.range(num_batch)*dim1, tf.reduce_prod(out_size, keepdims=True))
    base_y0 = base + y0*dim2
    base_y1 = base + y1*dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # use indices to lookup pixels in the flat image and restore
    # channels dim
    im_flat = tf.reshape(im, tf.stack([-1, channels],0))
    im_flat = tf.cast(im_flat, 'float32')
    Ia = tf.gather(im_flat, idx_a)
    Ib = tf.gather(im_flat, idx_b)
    Ic = tf.gather(im_flat, idx_c)
    Id = tf.gather(im_flat, idx_d)

    # and finally calculate interpolated values
    x0_f = tf.cast(x0, 'float32')
    x1_f = tf.cast(x1, 'float32')
    y0_f = tf.cast(y0, 'float32')
    y1_f = tf.cast(y1, 'float32')
    wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
    wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
    wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
    wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
    #output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id],name=opName)
    output_flat = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
    output = tf.reshape(output_flat,tf.shape(im),name=opName)

    return output


def pad_toroidal(X, w):
    """
    Pads the input tensor with toroidal ("circular") padding to induce periodic boundary conditions

    That is:
              DCDC
    AB  --\   BABA
    CD  --/   DCDC
              BABA

    Args:
        X: the tensor to pad
        w: the width of the padding

    Returns:
        X - padded version of input tensor

    """

    assert w is not None

    X_pad = tf.concat([X[:, -w::, :, :], X, X[:, 0:w, :, :]], 1)
    left_pad = tf.concat([X[:, -w::, -w::, :], X[:, :, -w::, :], X[:, 0:w, -w::, :]], 1)
    right_pad = tf.concat([X[:, -w::, 0:w, :], X[:, :, 0:w, :],  X[:, 0:w, 0:w, :]], 1)
    X_pad = tf.concat([left_pad, X_pad, right_pad], 2)

    return X_pad


def _repeat(x, n_repeats):
    rep = tf.transpose(
        tf.expand_dims(tf.ones(shape=n_repeats), 1), [1, 0])
    rep = tf.cast(rep, 'int32')
    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    return tf.reshape(x, [-1])


@tf.keras.utils.register_keras_serializable(package="Iteration")
def logits_loss(z_true, z_pred):
    """
    A cross-entropy loss that handles the expected Z shape and expects logits as input.
    """
    return K.mean(K.binary_crossentropy(z_true, z_pred, from_logits=True), axis=[3, 2, 1])


@tf.keras.utils.register_keras_serializable(package="Iteration")
def accuracy(z_true, z_pred):
    """
    Accuracy metric which requires all digits of z match.
    """
    return K.equal(K.sum(K.cast(K.equal(z_true, K.round(K.sigmoid(z_pred))), dtype='int32'), axis=-1), K.shape(z_true)[-1])


@tf.keras.utils.register_keras_serializable(package="Iteration")
def x_chan_mean_loss(z_true, x_stack):
    """
    Loss which penalizes changes in the mean value of each channel.
    """

    # Shift into concentration space
    x_stack = (x_stack + 1) / 2

    # Reproduce mean_concentration_delta to avoid having ops import anything
    c_mean_0 = tf.reduce_mean(x_stack[:, :, :, :, 0], axis=(1, 2))
    c_mean_1 = tf.reduce_mean(x_stack[:, :, :, :, -1], axis=(1, 2))
    c_delta = c_mean_1 - c_mean_0

    return tf.reduce_mean(tf.square(c_delta))


def x_similarity_loss(z_true, x_stack, epsilon=1e-3):
    """
    An RMSD / L2 norm loss for encouraging state-space similarity.
    Function signature is weird to support keras built-in training
    """

    # Normalize each sample in the reference to eliminate gradients towards degenerate zero-variance solutions.
    x_ref = x_stack[:, :, :, :, 0]
    x_ref = x_ref - tf.reduce_mean(x_ref, axis=(3, 2, 1), keepdims=True)
    x_ref = x_ref / (tf.math.reduce_std(x_ref, axis=(3, 2, 1), keepdims=True) + epsilon)

    x_1_delta = tf.sqrt(tf.reduce_mean((x_ref - x_stack[:, :, :, :, 1]) ** 2))

    if x_stack.shape[-1] > 2:
        # Handle case of mixed stochastic iterator or replication model
        x_2_delta = tf.sqrt(tf.reduce_mean((x_ref - x_stack[:, :, :, :, 2]) ** 2))
    else:
        x_2_delta = 0

    return x_1_delta + x_2_delta


def x_vs_t_similarity_loss(z_true, x_stack, epsilon=1e-15):
    """
    A spatiotemporal x-similarity loss which compares two input 5D X-vs-time tensors to a reference tensor.
    """
    x_ref = x_stack[:, :, :, :, :, 0]
    x_1 = x_stack[:, :, :, :, :, 1]
    x_2 = x_stack[:, :, :, :, :, 2]

    # # Normalize each sample/channel/timepoint in the reference to avoid degenerate zero-variance solutions.
    # x_means = tf.reduce_mean(x_ref, axis=(3, 2), keepdims=True)
    # x_ref = x_ref - x_means
    # x_ref = x_ref / (tf.math.reduce_std(x_ref, axis=(3, 2), keepdims=True) + epsilon)
    # x_ref = x_ref + x_means

    x_1_delta = tf.reduce_mean((x_ref - x_1) ** 2)
    x_2_delta = tf.reduce_mean((x_ref - x_2) ** 2)

    #var_delta = tf.reduce_mean((1 - tf.math.reduce_std(x_ref, axis=(3, 2), keepdims=True))**2)
    var_delta_1 = tf.reduce_mean(tf.maximum(1/2 - tf.math.reduce_std(x_1, axis=(3, 2), keepdims=True), 0)**2)
    var_delta_2 = tf.reduce_mean(tf.maximum(1/2 - tf.math.reduce_std(x_2, axis=(3, 2), keepdims=True), 0)**2)

    return x_1_delta + x_2_delta + .5*var_delta_1 + .5*var_delta_2 #10*var_delta


def x_corr_loss(z_true, x_stack, t=0):
    """
    Wrapper for using a pearson corr loss with our multi-output setup and keras builtin training.
    """
    if len(x_stack.shape) == 6:
        r01 = pearson_corr(x_stack[t, :, :, :, :, 0], x_stack[t, :, :, :, :, 1])
        if x_stack.shape[-1] > 2:
            r02 = pearson_corr(x_stack[t, :, :, :, :, 0], x_stack[t, :, :, :, :, 2])
        else:
            r02 = r01
    elif len(x_stack.shape) == 5:
        r01 = pearson_corr(x_stack[:, :, :, :, 0], x_stack[:, :, :, :, 1])
        if x_stack.shape[-1] > 2:
            r02 = pearson_corr(x_stack[:, :, :, :, 0], x_stack[:, :, :, :, 2])
        else:
            r02 = r01

    return -(r01 + r02) / 2


def pearson_corr(x, y):
    """
    Calculates the per-sample pearson correlation (PMCC) of the two input 4D tensors.
    Args:
        x: 4D tensor, 1st axis assumed to be batch axis
        y: 4D tensor, 1st axis assumed to be batch axis

    Returns:
        r: per-sample correlation, length equal to batch axis of x & y

    """
    mu_x = tf.reduce_mean(x, axis=(3, 2, 1), keepdims=True)
    mu_y = tf.reduce_mean(y, axis=(3, 2, 1), keepdims=True)
    std_x = tf.math.reduce_std(x, axis=(3, 2, 1))
    std_y = tf.math.reduce_std(y, axis=(3, 2, 1))
    n = tf.reduce_prod(x.shape[1::])

    cov = tf.reduce_sum( (x - mu_x) * (y - mu_y), axis=(3, 2, 1)) / tf.cast(n, x.dtype) # Biased cov but unimportant for our uses

    return cov / (std_x * std_y + 1e-16)


def asymmetric_x_corr_loss(z_true, x_stack):
    """
    Wrapper for using a pearson corr loss with out multi-output setup and keras builtin training.
    """
    r01 = tf.abs(1 - asymmetric_corr(x_stack[:, :, :, :, 0], x_stack[:, :, :, :, 1]))
    r02 = tf.abs(1 - asymmetric_corr(x_stack[:, :, :, :, 0], x_stack[:, :, :, :, 2]))
    return (r01 + r02) / 2


def asymmetric_corr(x, y):
    """
    Calculates an "asymmetric" version of Pearson's correlation which is normalized only by the variance of X
    """
    mu_x = tf.reduce_mean(x, axis=(3, 2, 1), keepdims=True)
    mu_y = tf.reduce_mean(y, axis=(3, 2, 1), keepdims=True)
    var_x = tf.math.reduce_variance(x, axis=(3, 2, 1))
    n = tf.reduce_prod(x.shape[1::])

    cov = tf.reduce_sum((x - mu_x) * (y - mu_y), axis=(3, 2, 1)) / tf.cast(n, x.dtype)  # Biased cov but unimportant for our uses

    return cov / var_x


def image_similarity_loss(z_true, x_stack, ref_image, chan_map):
    """
    Loss which applies RMSD similarity loss between X outputs and a reference image
    Args:
        z_true: z output of propagation model
        x_stack: (batch, h, w, c, output) X outputs of propagation model
        ref_image: (h, w, c) reference image
        chan_map:  (n_compare, 2) array mapping of channels to compare e.g.
            chan_map[i,0] in reference is compared to chan_map[i,1] in model output

    Returns:
        loss: RMSD loss
    """
    chan_map = np.array(chan_map)

    # Extract the selected channels from reference and model
    x_ref = tf.gather(ref_image, chan_map[:, 0], axis=2)

    x_model = tf.gather(x_stack[:, :, :, :, 0], chan_map[:, 1], axis=3)
    rmsd_stochastic = tf.sqrt(tf.reduce_mean((x_ref - x_model) ** 2))

    x_model = tf.gather(x_stack[:, :, :, :, 1], chan_map[:, 1], axis=3)
    rmsd_t = tf.sqrt(tf.reduce_mean((x_ref - x_model) ** 2))

    # We return an average of the stochastic batch and the t=T batch.
    return (rmsd_stochastic + rmsd_t) / 2


def image_correlation_loss(z_true, x_stack, ref_image, chan_map, i_out_compare):
    """
    Loss which applies pearson correlation similarity loss between X outputs and a reference image
    Args:
        z_true: z output of propagation model
        x_stack: (batch, h, w, c, output) X outputs of propagation model
        ref_image: (h, w, c) reference image
        chan_map:  (n_compare, 2) array mapping of channels to compare e.g.
            chan_map[i,0] in reference is compared to chan_map[i,1] in model output
        i_out_compare: Index of x output to compare reference image to

    Returns:
        loss: pearson correlation loss
    """
    chan_map = np.array(chan_map)

    # Extract the selected channels from reference and model
    x_ref = tf.gather(ref_image, chan_map[:, 0], axis=2)
    x_model = tf.gather(x_stack[:, :, :, :, i_out_compare], chan_map[:, 1], axis=3)

    x_ref = tf.expand_dims(x_ref, 0)  # Give the reference a batch axis so broadcasting works in corr func

    return pearson_corr(x_ref, x_model)


def bounded_batch_correlation_loss(z, x_stack, ref_image, chan_map):

    chan_map = np.array(chan_map)

    # Extract the selected channels from reference and model
    x_ref = tf.gather(ref_image, chan_map[:, 0], axis=2)
    x_model = tf.gather(x_stack, chan_map[:, 1], axis=3)
    # Treat the two outputs as a single batch. Too tired to think if a reshape would do this right.
    x_model = tf.concat([x_model[:, :, :, :, 0], x_model[:, :, :, :, 1]], axis=0)

    x_ref = tf.expand_dims(x_ref, 0)  # Give the reference a batch axis so broadcasting works in corr func

    mu_ref = tf.reduce_mean(x_ref)
    mu_model = tf.reduce_mean(x_model)
    std_ref = tf.math.reduce_std(x_ref)
    std_model = tf.maximum(tf.math.reduce_std(x_model), [0.1])

    cov = tf.reduce_mean((x_ref - mu_ref) * (x_model - mu_model))

    return cov / (std_ref * std_model)


def identity_kernel(shape, dtype):
    """
    Returns a kernel which corresponds to identity transform when used in convolution.
    Signature follows that of keras Initializers.
    """

    assert shape[0] % 2 == 1, "We expect an odd kernel size!"

    kernel = np.zeros(shape=shape)
    center = int((shape[1] - 1) / 2)
    for i_chan in range(shape[2]):
        kernel[center, center, i_chan, min(i_chan, shape[-1]-1)] = 1.0 # The min handles depthwise kernels

    return tf.convert_to_tensor(kernel, dtype=dtype)


def channel_averaging_kernel(shape, dtype, n_chan_average=1):
    """
    Returns a kernel which locally averages each channel independently and maps it to a single channel in the output.
    Signature follows that of keras Initializers.

    Args:
        n_chan_average: Number of channels to use locally averaging kernel for. Other channels will have identity
        kernel.
    """

    assert shape[0] % 2 == 1, "We expect an odd kernel size!"
    center = int((shape[1] - 1) / 2)
    kernel = np.zeros(shape=shape)
    for i_chan in range(shape[2]):
        if i_chan < n_chan_average:
            kernel[:, :, i_chan, min(i_chan, shape[-1]-1)] = 1.0 / np.prod(shape[0:2])
        else:
            kernel[center, center, i_chan, min(i_chan, shape[-1]-1)] = 1.0

    return tf.convert_to_tensor(kernel, dtype=dtype)


def laplacian_kernel(n_chan, sigma=0):
    """
    Returns a kernel for approximating the 2D laplacian operator via depthwise convolution
    Args:
        n_chan: number of channels in image to be filtered
        sigma: If 0, discrete laplacian operator kernel is returned. If > 0 then a laplacian-of-gaussian kernel with
            the specified sigma is returned (should really be >=1.0 for reasonable approximation).

    Returns:
        k: 3x3xn_chanx1 filter kernel

    """
    if sigma == 0:
        k = [[.25, .5, .25],
             [.5,  -3,  .5],
             [.25, .5, .25]]
    else:
        w = math.ceil(2*sigma)
        if w % 2 != 0:
            w = w + 1
        x, y = np.meshgrid(np.arange(-w, w+1), np.arange(-w, w+1))
        x2y2 = x ** 2 + y ** 2

        k = -1 / (math.pi * sigma ** 4) * (1 - x2y2 / (2 * sigma ** 2)) * np.exp(-x2y2 / (2 * sigma ** 2))

        k = k - np.mean(k)

    k = tf.convert_to_tensor(k, tf.float32)

    return tf.expand_dims(tf.tile(tf.expand_dims(k, -1), [1, 1, n_chan]), -1)


def gaussian_kernel_1d(sigma, w=None):

    if w is None:
        w = math.ceil(3*sigma)

    k = tf.math.exp(-tf.range(-w, w+1, dtype=tf.float32) ** 2 / (2 * sigma ** 2))
    return tf.cast(k / tf.reduce_sum(k), tf.float32)


def get_batch_trailing_fraction(z_true, x_stack, fraction):
    n = math.ceil(x_stack.shape[0] * fraction)
    return z_true[-n:], x_stack[-n:]


class MinMaxValueConstraint(Constraint):
    """
    Elementwise min-max value weight constraint
    """
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):

        return tf.clip_by_value(w, clip_value_min=self.min_value, clip_value_max=self.max_value)

    def get_config(self):

        return {'min_value': self.min_value,
                'max_value': self.max_value}
