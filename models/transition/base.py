"""
The abstract base class for transition models
"""
import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer


class TransitionLayer(Layer):
    """
    An abstract base class for "transition model" Layer objects.
    Transition models are models which propagate a state space X forward in time by one increment, e.g.
    X[t+1] = transition_model(X[t])
    """
    def __init__(self, d_t=1, **kwargs):
        """
        Args:
            d_t: time step size
            **kwargs: args to pass to parent __init__
        """
        super(TransitionLayer, self).__init__(**kwargs)

        self.d_t = d_t

    def call(self, x_t, i_t=0, seed=42, **kwargs):
        """
        Propagates x_t forward in time from step i_t to step i_t + 1
        Args:
            x_t: state to propagate
            i_t: time step (index) corresponding to x_t (NOT x_t_plus_1)
            seed: a seed which can be used for e.g. per-sample rather than per-call randomness.
            **kwargs:

        Returns:
            x_t_plus_1: the state at the next timepoint
        """
        raise NotImplementedError("This is an abstract base class and cannot be used in a model!")

    def get_config(self):
        config = super(TransitionLayer, self).get_config()
        config.update({'d_t': self.d_t})
        return config

    @classmethod
    def de_center_x(cls, x_t):
        """
        Shift x from standardized (-1, 1) range expected by e.g. mapping models  to (0, 1) range.
        """
        return (x_t + 1) / 2

    @classmethod
    def center_x(cls, x_t):
        """
        Shift x from (0, 1) range to standardized (-1, 1) range expected by e.g. mapping models.
        """
        return (x_t - .5) * 2

    def time(self, i_t):
        """
        Convert a time step index to physical time in seconds.
        Args:
            i_t: timepoint index (int).

        Returns:
            t: corresponding time in seconds (float).
        """
        return tf.cast(i_t, tf.float32) * self.d_t

    def log_free_parameters(self):
        """
        Logs any trainable parameters
        """
        pass
