"""
Models based on the Gray-Scott reaction system with varying feed schemes.
"""
import logging
import math

import numpy as np
import tensorflow as tf
from scipy.constants import Boltzmann as k_b
from tensorflow.python.keras.constraints import NonNeg
from tensorflow.python.keras.initializers.initializers_v2 import Constant
from tensorflow.python.keras.layers import Lambda, ReLU, serialize, deserialize

from models.transition.base import TransitionLayer
from ops import MinMaxValueConstraint, pad_toroidal, laplacian_kernel, gaussian_kernel_1d
from therm import diffusion_entropy_rate, irreversible_reaction_entropy_rate


@tf.keras.utils.register_keras_serializable(package="Iteration")
class GrayScottTransitionLayer(TransitionLayer):
    """
        A transition layer corresponding to the Gray-Scott reaction-diffusion model as described in [1]:

        U + 2V -> 3V
        V -> decay
        feed -> U

        [1] JE Pearson, Science 261, 189 (1993)
        """

    def __init__(self, init_diff_coef=(2e-1, 1e-1), init_feed_rate=.04, init_decay_rate=.06, fit_params=(),
                 diffusion_entropy_loss_weight=0.0, reaction_entropy_loss_weight=0.0, **kwargs):
        """
        Args:
            init_diff_coef: Initial diffusion coefficients for the substrate U and autocatalyst V respectively.
                Note that because our grid size=1 these are scaled relative to Pearson's values.
            init_feed_rate: Initial rate at which U is fed uniformly in the domain
            init_decay_rate: Initial rate at which V decays uniformly in the domain
            fit_params: An optional tuple of the parameters which should be fittable,
                e.g. ('init_diff_coef', 'init_decay_rate)
                or 'all' to fit all parameters, or () to fit none.
            diffusion_entropy_loss_weight: If non-zero, diffusion_entropy_rate loss term will be added using add_loss.
            reaction_entropy_loss_weight: If non-zero, reaction_entropy_rate loss term will be added using add_loss.

        NOTE: Do NOT use the initial or weight fields directly (e.g. don't use self.init_diff_coef or
            self.diff_coeffs) use the get methods instead e.g. DO use self.get_diff_coef(i_t) to ensure that e.g. temporal
            variability is properly handled for sub-classes.
        """
        super(GrayScottTransitionLayer, self).__init__(**kwargs)

        self.init_diff_coef = init_diff_coef
        self.init_feed_rate = init_feed_rate
        self.init_decay_rate = init_decay_rate

        self.diffusion_entropy_loss_weight = diffusion_entropy_loss_weight
        self.reaction_entropy_loss_weight = reaction_entropy_loss_weight

        if not isinstance(fit_params, (tuple, list)):
            fit_params = (fit_params,)
        for fit_param in fit_params:
            if fit_param not in ('init_diff_coef', 'init_feed_rate', 'init_decay_rate', 'all'):
                raise ValueError("Unrecognized parameter {} specified in fit_params!".format(fit_param))
        self.fit_params = fit_params

    def build(self, input_shape):

        self.diff_coeffs = self.add_weight(shape=2, initializer=Constant(value=self.init_diff_coef), name='diff_coeffs',
                                           #constraint=tf.keras.constraints.NonNeg(),
                                           constraint=MinMaxValueConstraint(min_value=.05, max_value=.2),
                                           trainable='init_diff_coef' in self.fit_params or 'all' in self.fit_params)

        self.feed_rate = self.add_weight(shape=1, initializer=Constant(value=self.init_feed_rate), name='feed_rate',
                                         constraint=NonNeg(),
                                         trainable='init_feed_rate' in self.fit_params or 'all' in self.fit_params)

        self.decay_rate = self.add_weight(shape=1, initializer=Constant(value=self.init_decay_rate), name='decay_rate',
                                          constraint=NonNeg(),
                                          trainable='init_decay_rate' in self.fit_params or 'all' in self.fit_params)

        self.x_shape = input_shape

    def call(self, x_t, i_t=0, seed=42, **kwargs):

        x_t = self.de_center_x(x_t)
        i_t = tf.ones_like(x_t[:, 0, 0, 0], dtype=tf.int32) * i_t  # Convert to (possibly unknown) batch-length vector
        x_t_plus_1, v_fwd = self.integrate(x_t, self.get_diff_coef(i_t), self.get_feed_rate(i_t, seed=seed),
                                           self.get_decay_rate(i_t), self.d_t)
        x_t_plus_1 = self.center_x(x_t_plus_1)

        diff_entropy_rate = tf.reduce_mean(diffusion_entropy_rate(x_t, self.get_diff_coef(i_t))) * self.d_t
        self.add_metric(diff_entropy_rate, name='diffusion_entropy_rate')
        if not self.diffusion_entropy_loss_weight == 0:
            self.add_loss(self.scale_diffusion_loss(diff_entropy_rate))

        rxn_entropy_rate = tf.reduce_mean(irreversible_reaction_entropy_rate(v_fwd)) * self.d_t
        self.add_metric(rxn_entropy_rate, name='reaction_entropy_rate')
        if not self.reaction_entropy_loss_weight == 0:
            self.add_loss(self.scale_reaction_loss(rxn_entropy_rate))

        return x_t_plus_1

    def scale_diffusion_loss(self, diff_entropy_rate):
        # We allow user-specified weighting and remove the factor of kb to bring it into order 1 magnitude.
        return tf.math.multiply(diff_entropy_rate, self.diffusion_entropy_loss_weight * 1/k_b,
                                name='diffusion_entropy_rate_loss')

    def scale_reaction_loss(self, rxn_entropy_rate):
        return tf.math.multiply(rxn_entropy_rate, self.reaction_entropy_loss_weight * 1/k_b,
                                name='reaction_entropy_rate_loss')

    def integrate(self, x_t, diff_coef, feed_rate, decay_rate, d_t):
        """
        Performs forward integration via Euler method.
        """

        del2_x = tf.nn.depthwise_conv2d(pad_toroidal(x_t, 1), laplacian_kernel(2), [1, 1, 1, 1], 'VALID')
        uv2 = x_t[:, :, :, 0] * x_t[:, :, :, 1] ** 2

        du = diff_coef[0] * del2_x[:, :, :, 0] - uv2 + feed_rate * (1 - x_t[:, :, :, 0])
        dv = diff_coef[1] * del2_x[:, :, :, 1] + uv2 - (feed_rate + decay_rate) * x_t[:, :, :, 1]

        dx = tf.stack([du, dv], axis=-1)

        x_t_plus_1 = x_t + d_t * dx
        # Clip any negative concentrations resulting from integration error, as well as enforcing a concentration cap
        x_t_plus_1 = tf.clip_by_value(x_t_plus_1, 0.0, 10.0)

        # Calculate and return the forward reaction rates of the autocatalytic and decay reactions
        v_fwd = [uv2, x_t[:, :, :, 1] * decay_rate]

        return x_t_plus_1, v_fwd
    """
    Override these methods to allow e.g. sample or time-dependent parameters. 
    """
    def get_diff_coef(self, i_t):
        return self.diff_coeffs

    def get_feed_rate(self, i_t, seed=None):
        return self.feed_rate

    def get_decay_rate(self, i_t):
        return self.decay_rate

    def get_config(self):
        config = super(GrayScottTransitionLayer, self).get_config()
        config.update({'init_diff_coef': self.init_diff_coef,
                       'init_feed_rate': self.init_feed_rate,
                       'init_decay_rate': self.init_decay_rate,
                       'fit_params': self.fit_params})
        return config

    def log_free_parameters(self):
        logging.info("Feed rate: {}".format(self.feed_rate.numpy()))
        logging.info("Decay rate: {}".format(self.decay_rate.numpy()))
        logging.info("Diffusion coefficients: {}".format(self.diff_coeffs.numpy()))


@tf.keras.utils.register_keras_serializable(package="Iteration")
class VariableFeedGrayScottTransitionLayer(GrayScottTransitionLayer):
    """
    A transition layer corresponding to the Gray-Scott reaction-diffusion model as described in [1], but with an
    arbitrary time-dependent feed rate.

    U + 2V -> 3V
    V -> decay
    feed -> U (k = F(t))

    [1] JE Pearson, Science 261, 189 (1993)
    """
    def __init__(self, feed_rate_func=Lambda(lambda t: .001 * t), **kwargs):
        """
        Args:
            feed_rate_func: a function of the form feed_rate_t = func(t) which returns the time-dependent *delta* in the
                feed rate. That is:
                 actual_feed_rate(t) = max(init_feed_rate + feed_rate_func(t), 0)

            Note that defining the feed rate  as a Keras Lambda layer can help with serialization/deserialization but
            it must be self-contained (not using variables from the scope it's defined in) or those variables must be
            defined in the scope it's deserialized in. Sub-classing this layer gives more portability.

        """
        super(VariableFeedGrayScottTransitionLayer, self).__init__(**kwargs)

        self._feed_rate_func = feed_rate_func

    def get_feed_rate(self, i_t, seed=None):
        feed_rate = self.feed_rate + self._feed_rate_func(self.time(i_t))
        return ReLU()(tf.reshape(feed_rate, (-1, 1, 1)))  # Clip zeros and return a shape suitable for broadcasting

    def get_config(self):
        config = super(VariableFeedGrayScottTransitionLayer, self).get_config()
        config.update({'feed_rate_func': serialize(self._feed_rate_func)})
        return config

    @classmethod
    def from_config(cls, config):
        config['feed_rate_func'] = deserialize(config['feed_rate_func'])
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="Iteration")
class SinusoidalFeedGrayScottTransitionLayer(GrayScottTransitionLayer):
    """
    A transition layer corresponding to the Gray-Scott reaction-diffusion model but with a
    sinusoidally time-dependent feed rate.
    """
    def __init__(self, period=100, amplitude=0.01, **kwargs):
        """
        Args:
            period: The period, in seconds, of the sinusoidal feed rate variation around the base feed rate.
            amplitude: The amplitude of the sinusoidal variation in feed rate.
        """
        super(SinusoidalFeedGrayScottTransitionLayer, self).__init__(**kwargs)

        self.period = period
        self.amplitude = amplitude

    def get_feed_rate(self, i_t, seed=None):
        feed_rate = self.feed_rate + self.amplitude * tf.math.sin(self.time(i_t) * 2.0 * math.pi / self.period)
        return ReLU()(tf.reshape(feed_rate, (-1, 1, 1)))  # Clip zeros and return a shape suitable for broadcasting

    def get_config(self):
        config = super(SinusoidalFeedGrayScottTransitionLayer, self).get_config()
        config.update({'period': self.period,
                       'amplitude': self.amplitude})
        return config


@tf.keras.utils.register_keras_serializable(package="Iteration")
class RandomFeedGrayScottTransitionLayer(GrayScottTransitionLayer):
    """
    A transition layer corresponding to the Gray-Scott reaction-diffusion model but with a
    sinusoidally time-dependent feed rate.
    """
    def __init__(self, amplitude=0.01, seed=None, **kwargs):
        """
        Args:
            amplitude: The amplitude of the variation in feed rate around the base feed rate
                (1/2 width of mean-zero uniform distribution added to base feed rate)
            seed: Random seed. If not None behavior will be identical *every call/batch* at a particular timepoint,
                (but different at each timepoint)
        """
        super(RandomFeedGrayScottTransitionLayer, self).__init__(**kwargs)

        self.amplitude = amplitude
        self.seed = seed
        if seed is not None:
            # We avoid issues with distributed compute / global seed setting by just storing a definitely-long-enough
            # sequence generated by numpy
            self._sequence = tf.cast(tf.convert_to_tensor(np.random.default_rng(self.seed).uniform(
                low=-self.amplitude, high=self.amplitude, size=(int(1e5),))), tf.float32)
        else:
            self._sequence = None

    def get_feed_rate(self, i_t, seed=None):
        i_t = tf.convert_to_tensor(i_t)
        if self.seed is not None:
            feed_rate = self.feed_rate + tf.gather(self._sequence, i_t)
        else:
            feed_rate = self.feed_rate + tf.random.uniform(shape=i_t.shape,
                                                           minval=-self.amplitude, maxval=self.amplitude)

        return ReLU()(tf.reshape(feed_rate, (-1, 1, 1)))  # Clip zeros and return a shape suitable for broadcasting

    def get_config(self):
        config = super(RandomFeedGrayScottTransitionLayer, self).get_config()
        config.update({'amplitude': self.amplitude,
                       'seed': self.seed})
        return config


@tf.keras.utils.register_keras_serializable(package="Iteration")
class GaussianGridFeedGrayScottTransitionLayer(GrayScottTransitionLayer):
    """
    A Gray-Scott reaction-diffusion transition layer which produces spatially and temporally localized feed in the form
    of random spacetime gaussians.

    The domain will be subdivided spatially into a grid, and then pulses of feed will be randomly generated at random
    grid positions and random times.

    """
    def __init__(self, n_grid_divisions=2, pulse_frequency=128, pulse_amplitude=0.1, pulse_sigma_t=64, max_n_t=8192,
                 constant_center=True, constant_baseline=False, **kwargs):
        """
        Args:
            n_grid_divisions: spatial domain will be divided into a grid with this many divisions along width & height
            pulse_frequency: The frequency (1/s) at which pulses of feed occur
            pulse_amplitude: The peak feed rate of the pulses.
            pulse_sigma_t: The sigma of the gaussian in the time direction.
            max_n_t: Maximum time*step* (not seconds) the model will be called at.
            constant_center: If true, a constant gaussian feed will be present at the center, with feed_rate amplitude.
            constant_baseline: If true, a uniform, constant baseline offset feed will be included (times the feed_rate
                param so fittable)
        """
        super(GaussianGridFeedGrayScottTransitionLayer, self).__init__(**kwargs)

        assert (max_n_t * self.d_t) % pulse_frequency == 0
        self.n_grid_divisions = n_grid_divisions
        self.pulse_frequency = pulse_frequency
        self.pulse_amplitude = pulse_amplitude
        self.pulse_sigma_t = pulse_sigma_t
        self.max_n_t = max_n_t
        self.constant_center = constant_center
        self.constant_baseline = constant_baseline

        self.pulse_frequency_it = int(pulse_frequency / self.d_t)  # Pulse frequency in timesteps
        self.n_pulses = int(self.max_n_t / self.pulse_frequency_it)

    def get_feed_rate(self, i_t, seed=42):
        # Re-generating these random values on each call with an input seed is our workaround for letting the
        # behavior be random between minibatches but still consistent over time within a minibatch.
        try:
            # Try this first as it's a workaround for allowing variable batch size
            batch_size = len(i_t)
        except:
            batch_size = self.x_shape[0]

        w = tf.cast(self.x_shape[1] / (self.n_grid_divisions-1), tf.int32)

        # Get the times and locations for each pulse
        pulse_t = tf.random.stateless_uniform((batch_size, self.n_pulses), (seed, seed), minval=0,
                                              maxval=self.max_n_t)
        pulse_x = tf.random.stateless_uniform((batch_size, self.n_pulses), (seed, seed + 1), minval=0,
                                              maxval=self.n_grid_divisions, dtype=tf.int32) * w
        pulse_y = tf.random.stateless_uniform((batch_size, self.n_pulses), (seed, seed + 2), minval=0,
                                              maxval=self.n_grid_divisions, dtype=tf.int32) * w

        # Use broadcasting to vectorize calculation of each pulse's contribution
        x, y = tf.meshgrid(tf.range(self.x_shape[1]), tf.range(self.x_shape[2]))
        x_p = tf.cast((tf.expand_dims(tf.expand_dims(x, 0), 0) - tf.expand_dims(tf.expand_dims(pulse_x, -1), -1)) ** 2,
                      tf.float32)
        y_p = tf.cast((tf.expand_dims(tf.expand_dims(y, 0), 0) - tf.expand_dims(tf.expand_dims(pulse_y, -1), -1)) ** 2,
                      tf.float32)
        t_p = tf.expand_dims(tf.expand_dims((tf.expand_dims(tf.cast(i_t, tf.float32), -1) - pulse_t) ** 2, -1), -1)
        s2xy = (tf.cast(w, tf.float32)/2) ** 2
        s2t = tf.cast(2*self.pulse_sigma_t**2, tf.float32)

        f = tf.reduce_sum(self.pulse_amplitude * tf.math.exp(-(x_p / s2xy + y_p / s2xy + t_p / s2t)), axis=1)
        if self.constant_center:
            x = tf.cast((tf.cast(self.x_shape[1]/2, tf.int32) - x) ** 2, tf.float32)
            y = tf.cast((tf.cast(self.x_shape[1]/2, tf.int32) - y) ** 2, tf.float32)
            f = f + self.feed_rate * tf.math.exp(-(x / s2xy + y / s2xy))

        if self.constant_baseline:
            f = f + self.feed_rate

        return f

    def get_config(self):
        config = super(GaussianGridFeedGrayScottTransitionLayer, self).get_config()
        config.update({'n_grid_divisions': self.n_grid_divisions,
                       'pulse_frequency': self.pulse_frequency,
                       'pulse_amplitude': self.pulse_amplitude,
                       'pulse_sigma_t': self.pulse_sigma_t,
                       'max_n_t': self.max_n_t,
                       'constant_center': self.constant_center,
                       'constant_baseline': self.constant_baseline})
        return config


@tf.keras.utils.register_keras_serializable(package="Iteration")
class GaussianPulseFeedGrayScottTransitionLayer(GrayScottTransitionLayer):
    """
    A Gray-Scott reaction-diffusion transition layer which produces spatially and temporally localized feed in the form
    of random spacetime gaussians.

    """
    def __init__(self, pulse_period=128, pulse_amplitude=0.1, pulse_sigma_xy=3, pulse_sigma_t=64, pulse_spacing=1,
                 max_n_t=8192, constant_baseline=True, **kwargs):
        """
        Args:
            n_grid_divisions: spatial domain will be divided into a grid with this many divisions along width & height
            pulse_period: The period (s) at which pulses of feed occur
            pulse_amplitude: The peak feed rate of the pulses.
            pulse_sigma_xy: The sigma of the gaussian in the xy directions.
            pulse_sigma_t: The sigma of the gaussian in the time direction.
            pulse_spacing: Integer minimum spacing between pulse centers. If > 1 centers will be on a grid with this spacing.
            max_n_t: Maximum time*step* (not seconds) the model will be called at.
            constant_baseline: If true, a uniform, constant baseline offset feed will be included (times the feed_rate
                param so fittable if that param is fittable)
        """
        super(GaussianPulseFeedGrayScottTransitionLayer, self).__init__(**kwargs)

        assert (max_n_t * self.d_t) % pulse_period == 0

        self.pulse_period = pulse_period
        self.pulse_amplitude = pulse_amplitude
        self.pulse_sigma_xy = pulse_sigma_xy
        self.pulse_sigma_t = pulse_sigma_t
        self.pulse_spacing = pulse_spacing
        self.max_n_t = max_n_t
        self.constant_baseline = constant_baseline

        self.pulse_period_it = int(pulse_period / self.d_t)  # Pulse frequency in timesteps
        self.n_pulses = int(self.max_n_t / self.pulse_period_it)
        self.k = gaussian_kernel_1d(self.pulse_sigma_xy)
        self.k = self.k / tf.reduce_max(self.k)  # Ensure that amplitudes are preserved

    def get_pulse_positions(self, batch_size, seed):
        # Re-generating these random values on each call with an input seed is our workaround for letting the
        # behavior be random between minibatches but still consistent over time within a minibatch.

        # Get the times and locations for each pulse
        pulse_shape = (batch_size, self.n_pulses)
        pulse_t = tf.random.stateless_uniform(pulse_shape, (seed, seed), minval=0, maxval=self.max_n_t, dtype=tf.int32)
        pulse_x = tf.random.stateless_uniform(pulse_shape, (seed, seed + 1), minval=0,
                                              maxval=int(self.x_shape[2] / self.pulse_spacing),
                                              dtype=tf.int32) * self.pulse_spacing
        pulse_y = tf.random.stateless_uniform(pulse_shape, (seed, seed + 2), minval=0,
                                              maxval=int(self.x_shape[1] / self.pulse_spacing),
                                              dtype=tf.int32) * self.pulse_spacing

        return pulse_t, pulse_x, pulse_y

    def get_feed_rate(self, i_t, seed=42):

        try:
            # Try this first as it's a workaround for allowing variable batch size
            batch_size = len(i_t)
        except:
            batch_size = self.x_shape[0]

        pulse_t, pulse_x, pulse_y = self.get_pulse_positions(batch_size, seed)
        # Get the amplitudes in the current time point
        t_dist = pulse_t - tf.reshape(tf.cast(i_t, tf.int32), (-1, 1))
        pulse_a = tf.math.exp(-t_dist ** 2 / (2*self.pulse_sigma_t ** 2)) * self.pulse_amplitude


        # Create delta functions at the center of each gaussian using sparse tensor
        pulse_ind = tf.stack([tf.repeat(tf.range(batch_size), self.n_pulses),             # batch ind
                              tf.reshape(pulse_y, (-1,)),                                    # y / h
                              tf.reshape(pulse_x, (-1,)),                                    # x / w
                              tf.tile(tf.range(self.n_pulses), (batch_size,))], axis=1)   # pulse ind
        f = tf.sparse.SparseTensor(tf.cast(pulse_ind, tf.int64), tf.reshape(tf.cast(pulse_a, tf.float32), (-1,)),
                                   (batch_size,) + self.x_shape[1:3] + (self.n_pulses,))

        # Sum any overlapping pulse's amplitude
        f = tf.sparse.reduce_sum(f, axis=-1, keepdims=True, output_is_sparse=False)

        # A (spatially) separable gaussian convolution converts these delta functions to gaussians
        f = tf.nn.conv2d(f, tf.reshape(self.k, (-1, 1, 1, 1)), 1, "SAME")
        f = tf.nn.conv2d(f, tf.reshape(self.k, (1, -1, 1, 1)), 1, "SAME")
        f = tf.squeeze(f)

        if self.constant_baseline:
            f = f + self.feed_rate

        return f

    def get_config(self):
        config = super(GaussianPulseFeedGrayScottTransitionLayer, self).get_config()
        config.update({'pulse_period': self.pulse_period,
                       'pulse_amplitude': self.pulse_amplitude,
                       'pulse_sigma_xy': self.pulse_sigma_xy,
                       'pulse_sigma_t': self.pulse_sigma_t,
                       'pulse_spacing': self.pulse_spacing,
                       'max_n_t': self.max_n_t,
                       'constant_baseline': self.constant_baseline})
        return config