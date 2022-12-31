"""
Models for driving reaction diffusion systems away from equilibrium.
"""
import logging
import numpy as np
import math

import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow.keras.constraints import NonNeg
from layers import GaussianFilterLayer
from ops import MinMaxValueConstraint


class BaseDriveLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BaseDriveLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        Returns the drive dx_dt, the additional change in state due to the drive.
        Args:
            x_t: current state, (batch, height, width, n_species) tensor
            i_t: current time. Not used, for compatibility.
            **kwargs:

        Returns:
            dx_dt: The time derivative component due to the drive.
        """
        raise NotImplementedError("Abstract base class!")

    def log_free_parameters(self):
        """
        Logs any parameters which are optimizable/trainable
        """
        pass


@tf.keras.utils.register_keras_serializable(package="Iteration")
class ZeroDriveLayer(BaseDriveLayer):
    """
    A do-nothing drive to serve as a placeholder for un-driven systems and avoid using Lambda / if statements in .call()
    """
    def __init__(self, **kwargs):
        super(ZeroDriveLayer, self).__init__(**kwargs)

    def call(self, x_t, i_t=0):
        return 0.0


@tf.keras.utils.register_keras_serializable(package="Iteration")
class FlowDriveLayer(BaseDriveLayer):
    """
    A layer that models the drive of a flow reactor, which maintains a nonequilibrium state via the continuous inflow
    of medium with reactant(s) at a specified fixed concentration, as well as outflow.
    """
    def __init__(self, init_feed_conc, init_flow_rate, fit_feed_conc=False, fit_flow_rate=False, **kwargs):
        """
        Args:
            init_feed_conc: n_species length vector of initial values for the concentrations of each species in the inflow in mol/L
            init_flow_rate: scalar initial value of flow rate in units of the inverse residence time or 1/s
            fit_feed_conc: If true, the feed concentration will be a fittable (trainable) parameter
            fit_flow_rate:  If true, the flow rate will be a fittable (trainable) parameter
        """
        super(FlowDriveLayer, self).__init__(**kwargs)

        self.init_feed_conc = np.array(init_feed_conc, dtype=np.float32)
        self.init_flow_rate = init_flow_rate
        self.fit_feed_conc = fit_feed_conc
        self.fit_flow_rate = fit_flow_rate

    def build(self, input_shape):
        self.feed_conc = self.add_weight(shape=len(self.init_feed_conc), initializer=Constant(value=self.init_feed_conc),
                                         name='feed_conc', constraint=NonNeg(), trainable=self.fit_feed_conc)
        self.flow_rate = self.add_weight(shape=1, initializer=Constant(value=self.init_flow_rate), name='flow_rate',
                                         constraint=MinMaxValueConstraint(1e-2, 1), trainable=self.fit_flow_rate)

    def call(self, x_t, i_t=0, **kwargs):

        return self.flow_rate * (tf.reshape(self.feed_conc, (1, 1, 1, -1)) - x_t)

    def get_config(self):
        config = super(FlowDriveLayer, self).get_config()
        config.update({'init_feed_conc': self.init_feed_conc,
                       'init_flow_rate': self.init_flow_rate,
                       'fit_feed_conc': self.fit_feed_conc,
                       'fit_flow_rate': self.fit_flow_rate})
        return config

    def log_free_parameters(self):
        logging.info("Flow rate: {}".format(self.flow_rate.numpy()))
        logging.info("Feed concentrations: {}".format(self.feed_conc.numpy()))


@tf.keras.utils.register_keras_serializable(package="Iteration")
class NoisyFlowDriveLayer(FlowDriveLayer):

    def __init__(self, noise_amplitude=.1, filter_sigma=1.0, **kwargs):
        """
        A flow reactor style drive but with spatially-variable random gaussian noise in the flow rate.
        The noise is low-pass filtered to avoid introducing spatial discontinuities.

        Args:
            noise_amplitude: Noise will be added from a uniform(-amp, amp) distribution before filtering.
            filter_sigma: Sigma of the gaussian low-pass spatial filter applied to the noise. 0=no filtering.
            **kwargs: Arguments to pass on to FlowDriveLayer
        """
        super(NoisyFlowDriveLayer, self).__init__(**kwargs)

        self.noise_amplitude = noise_amplitude
        self.filter_sigma = filter_sigma

    def call(self, x_t, i_t=0, **kwargs):

        flow_rate = self.flow_rate + tf.random.uniform(shape=x_t.shape,
                                                       minval=-self.noise_amplitude,
                                                       maxval=self.noise_amplitude)
        # Suppress any negative flow rates.
        flow_rate = tf.where(flow_rate < 0.0, 0.0, flow_rate)

        if self.filter_sigma > 0:
            flow_rate = GaussianFilterLayer(self.filter_sigma, math.ceil(3*self.filter_sigma))(flow_rate)

        return flow_rate * (tf.reshape(self.feed_conc, (1, 1, 1, -1)) - x_t)

    def get_config(self):
        config = super(NoisyFlowDriveLayer, self).get_config()
        config.update({'noise_amplitude': self.noise_amplitude,
                       'filter_sigma': self.filter_sigma})
        return config


@tf.keras.utils.register_keras_serializable(package="Iteration")
class ConstantSynthesisDriveLayer(BaseDriveLayer):
    """
    A layer that models the drive of (a) species being synthesized at a constant mol/s rate.
    """
    def __init__(self, init_synth_rate, fit_synth_rate=False, **kwargs):
        """
        Args:
            init_synth_rate: n_species length vector of initial values for the synthesis rate of each species in mol/s
            fit_synth_rate: If true, the synthesis rate will be a fittable (trainable) parameter
        """
        super(ConstantSynthesisDriveLayer, self).__init__(**kwargs)
        self.init_synth_rate = np.array(init_synth_rate, dtype=np.float32)
        self.fit_synth_rate = fit_synth_rate

    def build(self, input_shape):
        self.synth_rate = self.add_weight(shape=len(self.init_synth_rate),
                                          initializer=Constant(value=self.init_synth_rate),
                                          name='synth_rate', constraint=NonNeg(), trainable=self.fit_synth_rate)

    def call(self, x_t, i_t=0, **kwargs):
        """
        Returns the drive dx_dt, the additional change in state due to the drive.
        Args:
            x_t: current state, (batch, height, width, n_species) tensor
            i_t: current time. Not used, for compatibility.
            **kwargs:

        Returns:
            dx_dt: The time derivative component due to the drive.
        """
        return tf.reshape(self.synth_rate, (1, 1, 1, -1))

    def get_config(self):
        config = super(ConstantSynthesisDriveLayer, self).get_config()
        config.update({'init_synth_rate': self.init_synth_rate,
                       'fit_synth_rate': self.fit_synth_rate})
        return config

    def log_free_parameters(self):
        logging.info("Synthesis rates: {}".format(self.synth_rate.numpy()))
