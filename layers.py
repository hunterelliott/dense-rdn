"""
Custom keras utility layers for various models
"""

import pickle
import codecs
import logging

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Layer, serialize, deserialize, Cropping2D
from tensorflow.keras.initializers import Constant
from tensorflow_addons.image import interpolate_bilinear, gaussian_filter2d
from domains import get_domain_padding

from ops import MinMaxValueConstraint


def iteration_layer(layer_name='IterationLayer', transition_model=None, n_t=1, **kwargs):
    return globals()[layer_name](transition_model, n_t, **kwargs)


@tf.keras.utils.register_keras_serializable(package="Iteration")
class IterationLayerBase(Layer):
    """
    Abstract base class for iteration layers
    """
    def __init__(self, transition_model, n_t, x_output_period=1, seed=None, meta_loss_func=None, **kwargs):
        """
        Creates a layer which propagates a state X through time by iteratively composing an input transition model.
        Args:
            transition_model: model which will be composed, maps X_t to X_t+1
            n_t: number of timesteps to iterate over
            x_output_period: how often in time to output X. larger values will decrease memory usage.
            seed: A random seed which can be fed to the transition model to allow randomness which varies per
                time series rather than per time point.
            meta_loss_func: A function which defines a "meta loss" which is dependent on transition model losses, e.g.:
                meta_loss = meta_loss_func(transition_model.losses)
                self.add_loss(meta_loss)
            **kwargs: additional arguments to pass to parent class
        """

        assert n_t % x_output_period == 0, "n_t must be divisible by x_output_period!"

        self.transition_model = transition_model
        self.n_t = n_t
        self.x_output_period = x_output_period
        self.seed = seed
        self.meta_loss_func = meta_loss_func

        if seed is None:
            self.rng = tf.random.get_global_generator()
        else:
            self.rng = tf.random.Generator.from_seed(self.seed)

        super(IterationLayerBase, self).__init__(**kwargs)

    def call(self, x_t, **kwargs):
        raise NotImplementedError("This is an abstract base class and cannot be used in a model!")

    def add_meta_loss(self):
        if self.meta_loss_func is not None:
            loss_val = self.meta_loss_func(self.transition_model.losses, self.n_t)
            self.add_loss(loss_val)
            self.add_metric(loss_val, name='meta_loss')

    def get_config(self):
        config = super(IterationLayerBase, self).get_config()
        config.update({"transition_model": serialize(self.transition_model),
                       "n_t": self.n_t,
                       "x_output_period": self.x_output_period})
        return config

    @classmethod
    def from_config(cls, config):
        transition_model = deserialize(config['transition_model'])
        config['transition_model'] = transition_model
        return cls(**config)

    def compute_output_shape(self, input_shape):

        return (int(self.n_t / self.x_output_period) + 1,) + input_shape


@tf.keras.utils.register_keras_serializable(package="Iteration")
class IterationLayer(IterationLayerBase):
    """
    The standard iteration layer class concrete implementation.
    """
    def iterate(self, x_0):

        x_t = x_0
        x_vs_t = [x_0]
        # Create and pass a seed to allow per-iteration (rather than per-call) randomness in transition models.
        seed = self.rng.uniform_full_int((), dtype=tf.int32)

        for i_t in range(self.n_t):
            x_t = self.transition_model(x_t, i_t=i_t, seed=seed)
            if (i_t + 1) % self.x_output_period == 0:
                x_vs_t.append(x_t)

        self.add_meta_loss()

        return tf.stack(x_vs_t)

    def call(self, x_0, **kwargs):
        """

        Args:
            x_0: X at t=0

        Returns:
            X_vs_t: a ndims+1 tensor of X at all timepoints, where the first dimension is time.
            Note that there will be n_t / x_output_period + 1 timepoints, as X_0 is included.

        """

        return self.iterate(x_0)


@tf.keras.utils.register_keras_serializable(package="Iteration")
class SymbolicIterationLayer(IterationLayerBase):
    """
    This layer uses the symbolic loop in tf.while with memory swapping to iteratively compose an input transition model
    while decoupling timescale from GPU memory overhead.
    """
    def iterate(self, x_0):
        n_t_blocks = int(self.n_t / self.x_output_period)
        x_vs_t = [x_0]
        x_t = x_0
        i_t = tf.constant(0, dtype=tf.int32)
        # Create and pass a seed to allow per-iteration (rather than per-call) randomness in transition models.
        seed = self.rng.uniform_full_int((), dtype=tf.int32)

        for i_block in range(n_t_blocks):
            condition = lambda x, t, s: t < (i_block + 1) * self.x_output_period

            x_t, i_t, seed = tf.while_loop(condition, self.loop_body, [x_t, i_t, seed],
                                           parallel_iterations=1, swap_memory=True)

            x_vs_t.append(x_t)

        self.add_meta_loss()

        return tf.stack(x_vs_t)

    def call(self, x_0, **kwargs):
        """

        Args:
            x_0: X at t=0

        Returns:
            X_vs_t: a ndims+1 tensor of X at all timepoints, where the first dimension is time.
            Note that there will be n_t / x_output_period + 1 timepoints, as X_0 is included.

        """

        return self.iterate(x_0)

    def loop_body(self, x_t, i_t, seed):

        x_t = self.transition_model(x_t, i_t=i_t, seed=seed)

        return x_t, i_t + 1, seed


@tf.keras.utils.register_keras_serializable(package="Iteration")
class StochasticOutputIterationLayer(IterationLayer):
    def __init__(self, transition_model, n_t, **kwargs):
        """
        This layer iteratively composes an input transition model to propagate X through time, outputting a random
        sampling of Xs from this time series.
        Args:
            transition_model: model which will be composed, maps X_t to X_t+1
            n_t: number of timesteps to iterate over
            x_output_period: how often in time to output X. This will cause the random Xs to be sub-sampled in time.
            **kwargs: additional arguments to pass to parent class
        """
        self.i_t_sample = None
        super(StochasticOutputIterationLayer, self).__init__(transition_model, n_t, **kwargs)

    def sample_batch_from_time(self, x_vs_t):
        # Choose a random time to output X for each sample in the batch.
        batch_size = x_vs_t[0].shape[0]
        t_rand_sample = tf.random.uniform(shape=[batch_size], dtype=tf.int32, minval=0,
                                          maxval=int(self.n_t / self.x_output_period) + 1)
        batch_ind = tf.range(0, batch_size)
        self.i_t_sample = tf.stack([t_rand_sample, batch_ind], axis=1)
        x_random = tf.gather_nd(x_vs_t, self.i_t_sample)

        return x_random

    def call(self, x_0, **kwargs):
        """

        Args:
            x_0: X at t=0

        Returns:
            X_sample: a batch of X uniformly randomly sampled over all possible timepoints.

        """

        # We have to deterministically output Xs and then randomly sub-sample a batch from them to allow graph-mode
        # execution and the associated performance increase.
        x_vs_t = self.iterate(x_0)

        return self.sample_batch_from_time(x_vs_t)

    def compute_output_shape(self, input_shape):

        return input_shape


@tf.keras.utils.register_keras_serializable(package="Iteration")
class MixedStochasticOutputIterationLayer(StochasticOutputIterationLayer):
    def __init__(self, transition_model, n_t, **kwargs):
        """
        This layer iteratively composes an input transition model to propagate X through time, outputting a batch at
        X_0, a batch randomly sampled from [X_1, ..., X_t-1] and a batch from X_T

        Args:
            transition_model: model which will be composed, maps X_t to X_t+1
            n_t: number of timesteps to iterate over
            x_output_period: how often in time to output X. This will cause the random Xs to be sub-sampled in time.
            **kwargs: additional arguments to pass to parent class
        """
        self.i_t_sample = None
        super(MixedStochasticOutputIterationLayer, self).__init__(transition_model, n_t, **kwargs)

    def call(self, x_0, **kwargs):
        """

        Args:
            x_0: X at t=0

        Returns:
            [X_0, X_sample, X_t]: a batch of X_0, X uniformly randomly sampled from 1..T-1, and X_T.

        """

        x_vs_t = self.iterate(x_0)

        x_end = x_vs_t[-1]
        x_random = self.sample_batch_from_time(x_vs_t)

        return [x_0, x_random, x_end]

    def compute_output_shape(self, input_shape):

        return [input_shape, input_shape, input_shape]


@tf.keras.utils.register_keras_serializable(package="Iteration")
class ReplicationOutputLayer(Layer):
    """
    A layer for extracting the sub-domains corresponding to the parent and two daughters from the full X domains,
    possibly including a buffer region e.g. for x-similarity losses.
    """
    def __init__(self, replication_domain, include_buffer=False, **kwargs):
        """
        Args:
            replication_domain: The ReplicationDomain object describing the optimization domain
            include_buffer: If true, the output will include the buffer region around each daughter X.
            **kwargs:
        """
        self.replication_domain = replication_domain
        self.include_buffer = include_buffer

        super(ReplicationOutputLayer, self).__init__(**kwargs)

    def call(self, x_full_0, x_full_t, **kwargs):
        """

        Args:
            x_full_0: full X domain at t=0.
            x_full_t: full X domain at t=T

        Returns:
            (x_parent, x_daughter_1, x_daughter_2)
        """
        if self.include_buffer:
            x_b = self.replication_domain.buffer
        else:
            x_b = 0

        # The parent, from the center at t=0
        parent_pad = get_domain_padding(self.replication_domain.full_domain,
                                        self.replication_domain.parent_domain, buffer=x_b)
        if len(x_full_0.shape) > 4:  # Handle the x-vs-t output case
            x_parent = tf.map_fn(Cropping2D(parent_pad), x_full_0, fn_output_signature=tf.float32)
        else:
            x_parent = Cropping2D(parent_pad)(x_full_0)

        # The daughters, side by side at t=T
        d1_pad = get_domain_padding(self.replication_domain.full_domain,
                                    self.replication_domain.daughter_domains[0], buffer=x_b)
        if len(x_full_t.shape) > 4:
            x_daughter_1 = tf.map_fn(Cropping2D(d1_pad), x_full_t, fn_output_signature=tf.float32)
        else:
            x_daughter_1 = Cropping2D(d1_pad)(x_full_t)

        d2_pad = get_domain_padding(self.replication_domain.full_domain,
                                    self.replication_domain.daughter_domains[1], buffer=x_b)
        if len(x_full_t.shape) > 4:
            x_daughter_2 = tf.map_fn(Cropping2D(d2_pad), x_full_t, fn_output_signature=tf.float32)
        else:
            x_daughter_2 = Cropping2D(d2_pad)(x_full_t)

        return x_parent, x_daughter_1, x_daughter_2

    def get_config(self):
        config = super(ReplicationOutputLayer, self).get_config()
        config.update({"replication_domain": codecs.encode(pickle.dumps(self.replication_domain), "base64").decode(),
                       "include_buffer": self.include_buffer})
        return config

    @classmethod
    def from_config(cls, config):
        replication_domain = pickle.loads(codecs.decode(config['replication_domain'].encode(), "base64"))
        config['replication_domain'] = replication_domain
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="Iteration")
class DisplacementLayer(Layer):
    """
    A layer which uses an input vector field to displace the input tensor by interpolating at positions offset by the
    vector field. (This effectively makes the displacement field "reversed" in the sense that it points backwards in
    time rather than forward)
    """
    def __init__(self, **kwargs):
        super(DisplacementLayer, self).__init__(**kwargs)

    def call(self, x_t, uv, **kwargs):
        """

        Args:
            x_t: the (batch, w, h, c) 4D-tensor to displace
            uv: the (batch, w, h, 2) tensor of x-y reverse-displacement vectors.

        Returns:
            x_t_plus_1: the "displaced" by interpolating the previous timepoint, same shape as x_t
        """

        # These vectors are displacements so we convert to absolute interpolation coords first
        [x_coord, y_coord] = tf.meshgrid(range(x_t.shape[2]), range(x_t.shape[1]))
        x_coord = tf.cast(x_coord, tf.float32) + uv[:, :, :, 0]
        y_coord = tf.cast(y_coord, tf.float32) + uv[:, :, :, 1]
        x_t_plus_1 = interpolate_bilinear(x_t, tf.concat((tf.expand_dims(tf.reshape(x_coord, (x_t.shape[0], -1)), -1),
                                                          tf.expand_dims(tf.reshape(y_coord, (x_t.shape[0], -1)), -1)),
                                                         axis=2), indexing='xy')
        x_t_plus_1 = tf.reshape(x_t_plus_1, x_t.shape)

        return x_t_plus_1


@tf.keras.utils.register_keras_serializable(package="Iteration")
class RandomDaughterCenteringLayer(Layer):
    """
    A layer which circularly permutes the input X domain to center one of the daughters, randomly selecting between the
    left or right daughter.

    Note: If using in a distribution strategy you must first call tf.random.get_global_generator() outside of the
    distribution scope.
    """
    def __init__(self, x_daughter_offset, g=None, **kwargs):
        """
        Args:
            x_daughter_offset: The number of pixels the input domain must be shifted along the X dimension (2nd spatial
                dimension) such that one of the daughters will be centered.
        """
        self.x_daughter_offset = x_daughter_offset
        if g is None:
            self.g = tf.random.get_global_generator()

        super(RandomDaughterCenteringLayer, self).__init__(**kwargs)

    def call(self, x, **kwargs):

        shift = (self.g.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32) * 2) - 1
        shift = shift * self.x_daughter_offset

        return tf.roll(x, shift=shift, axis=2)

    def get_config(self):
        config = super(RandomDaughterCenteringLayer, self).get_config()
        config.update({"x_daughter_offset": self.x_daughter_offset,
                       "g": None})
        return config


@tf.keras.utils.register_keras_serializable(package="Iteration")
class GaussianFilterLayer(Layer):
    """
    Layer for performing gaussian filtering on images. Just a wrapper around TF addon function to make it play nice
    with Keras serialization, enable changes in implementation e.g. separability etc.
    """
    def __init__(self, sigma, w, **kwargs):
        """

        Args:
            sigma: the standard deviation of the gaussian kernal
            w: size of kernel in pixels.
        """

        self.sigma = sigma
        self.w = w

        super(GaussianFilterLayer, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        # Make sure we aren't accidentally doing 3D filtering...
        assert len(x.shape) == 4, "Input tensor must be (batch, h, w, chan) 4D!"

        return gaussian_filter2d(x, self.w, self.sigma)

    def get_config(self):
        config = super(GaussianFilterLayer, self).get_config()
        config.update({"sigma": self.sigma,
                       "w": self.w})
        return config


@tf.keras.utils.register_keras_serializable(package="Iteration")
class RescalingLayer(Layer):
    """
    Simple layer for a learned rescaling of the input which somehow doesn't exist already.

    """
    def __init__(self, init_scale=1.0, scale_range=(.1, 10), **kwargs):
        """
        Args:
            init_scale: Scalar to initialize the scaling to.
            scale_range: tuple with upper and lower bound for scaling.
        """
        self.init_scale = init_scale
        self.scale_range = scale_range

        super(RescalingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(shape=1, initializer=Constant(value=self.init_scale), name='scale', trainable=True,
                                     constraint=MinMaxValueConstraint(min_value=self.scale_range[0],
                                                                      max_value=self.scale_range[1]))

    def call(self, x, **kwargs):

        return x * self.scale

    def get_config(self):
        config = super(RescalingLayer, self).get_config()
        config.update({"init_scale": self.init_scale,
                       'scale_range': self.scale_range})
        return config


@tf.keras.utils.register_keras_serializable(package="Iteration")
class ChannelOffsetLayer(Layer):
    """
    Simple layer for a learned offset of a specific channel.
    (Assumes that the last dimension is the channel dimension)
    """
    def __init__(self, i_chan=None, init_offset=0, min_value=0, max_value=1.0, **kwargs):
        """
        Args:
            i_chan: Which channel(s) to add offset to
            init_offset: Scalar to initialize the offset(s) to
            min_value: lower bound for offset
            max_value: upper bound for offset
        """
        self.init_offset = init_offset
        self.i_chan = np.array(i_chan)
        self.n_chan = self.i_chan.size
        self.min_value = min_value
        self.max_value = max_value

        super(ChannelOffsetLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.offset = self.add_weight(shape=self.n_chan, initializer=Constant(value=self.init_offset),
                                      name='offset', trainable=True,
                                      constraint=MinMaxValueConstraint(min_value=self.min_value,
                                                                       max_value=self.max_value))

    def call(self, x, **kwargs):
        # We use a sparse tensor to allow only a subset of the offsets to be learnable
        offset_vec_shape = [1 for dim in x.shape]
        offset_vec_shape[-1] = x.shape[-1]
        indices = np.zeros(shape=(self.n_chan, len(offset_vec_shape)))
        indices[:, -1] = self.i_chan
        offset_vec = tf.sparse.SparseTensor(indices=indices, values=self.offset, dense_shape=offset_vec_shape)
        # But then we need to convert to dense to allow broadcasting. Sigh.
        offset_vec = tf.sparse.to_dense(offset_vec)

        # Use broadcasting to offset each channel
        return x + offset_vec

    def get_config(self):
        config = super(ChannelOffsetLayer, self).get_config()
        config.update({"init_offset": self.init_offset})
        config.update({"i_chan": self.i_chan,
                       'min_value': self.min_value,
                       'max_value': self.max_value})
        return config


@tf.keras.utils.register_keras_serializable(package="Iteration")
class ClippingLayer(Layer):
    """
    Simple layer for enforcing a range by clipping
    (but avoiding lamba layer use and (de)serialization problems)

    """
    def __init__(self, out_range=(-1.0, 1.0), **kwargs):
        """
        Args:
            out_range: Tuple specifying range to clip to
        """
        self.out_range = out_range
        super(ClippingLayer, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        return tf.clip_by_value(x, clip_value_min=self.out_range[0],
                                   clip_value_max=self.out_range[1])

    def get_config(self):
        config = super(ClippingLayer, self).get_config()
        config.update({"out_range": self.out_range})
        return config
