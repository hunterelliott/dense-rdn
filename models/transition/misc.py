import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.initializers.initializers_v2 import Constant
from tensorflow.python.keras.layers import Lambda, Conv2D, Concatenate

from models.transition.base import TransitionLayer
from ops import pad_toroidal, laplacian_kernel


@tf.keras.utils.register_keras_serializable(package="Iteration")
class BrusselatorTransitionLayer(TransitionLayer):
    """
    Simple 2D reaction-diffusion Brusselator layer implementation
    """
    def __init__(self, a=1.0, b=2.2, d_x=1.0, d_y=8.0, **kwargs):
        super(BrusselatorTransitionLayer, self).__init__(**kwargs)

        self.a = a
        self.b = b
        self.d_x = d_x
        self.d_y = d_y

    def call(self, x_t, i_t=0, seed=42, **kwargs):

        x_t = self.de_center_x(x_t)

        del2 = tf.nn.depthwise_conv2d(pad_toroidal(x_t, 1), laplacian_kernel(2), [1, 1, 1, 1], 'VALID')

        x2y = x_t[:, :, :, 0] ** 2 * x_t[:, :, :, 1]
        dx = self.a - (self.b + 1) * x_t[:, :, :, 0] + x2y + del2[:, :, :, 0] * self.d_x
        dy = self.b * x_t[:, :, :, 0] - x2y + del2[:, :, :, 1] * self.d_y

        d_dt = tf.stack([dx, dy], axis=-1)

        x_t_plus_1 = x_t + self.d_t * d_dt

        return self.center_x(x_t_plus_1)

    def get_config(self):
        config = super(BrusselatorTransitionLayer, self).get_config()
        config.update({'a': self.a,
                       'b': self.b,
                       'd_x': self.d_x,
                       'd_y': self.d_y})
        return config


def coupled_lorenz_transition_model(x_shape,
                                    epsilon=0.005,
                                    sigma=10.0,
                                    r=60.0,
                                    b=8.0/3.0):
    """
    Defines a 2D array of coupled Lorenz systems, with the deefault parameters corresponding to a chaotic regime.
    This is modelled after the binary coupled lorenz system described in [1], but with each system coupled to its 8
    neighbors in the array.

    Args:
        x_shape: shape of 2D array of systems
        epsilon: step size
        sigma, r, b: nondimensionalized parameters of lorenz system.

    Returns:
        model: A keras Model object describing the system.

    References:
        [1] Strogatz, Steven H. Nonlinear dynamics and chaos. CRC press, 2018, pp. 346
    """
    # X is a 2D array of Lorenz systems each described by a (u, v, w) tuple
    x_t = Input(shape=x_shape)

    # Separate state variable arrays for convenience
    U = Lambda(lambda x: K.expand_dims(x[:, :, :, 0]))(x_t)
    V = Lambda(lambda x: K.expand_dims(x[:, :, :, 1]))(x_t)
    W = Lambda(lambda x: K.expand_dims(x[:, :, :, 2]))(x_t) + 2.6  # Correct for non-zero ~mean but ~zero-centered state

    # Use local average of U as "transmitter" - input for local coupling, using toroidal boundary conditions
    U_pad = Lambda(lambda x: pad_toroidal(x, 1))(U)
    U_avg = Conv2D(1, 3, padding='valid', kernel_initializer=Constant(value=1.0/9.0), trainable=False)(U_pad)

    # Calculate temporal derivatives of state variable arrays

    # This appears to be necessary to avoid custom layer - just convert to custom layer?
    def dUdt(uv, sigma=sigma):
        u, v = uv
        return sigma * (v - u)

    def dVdt(uvw, r=r):
        u, v, w = uvw
        return r * u - v - 20.0 * u * w

    def dWdt(uvw, b=b):
        u, v, w = uvw
        return 5.0 * u * v - b * w

    delta_u = Lambda(dUdt)([U, V])
    delta_v = Lambda(dVdt)([U_avg, V, W])
    delta_w = Lambda(dWdt)([U_avg, V, W])

    # Update state arrays
    def update_state(y_delta_y, epsilon=epsilon):
        y, delta_y = y_delta_y
        return y + delta_y * epsilon

    u_prime = Lambda(update_state)([U, delta_u])
    v_prime = Lambda(update_state)([V, delta_v])
    w_prime = Lambda(update_state)([W, delta_w]) - 2.6  # Correct for non-zero ~mean to give ~ zero-centered state space

    x_t_plus_1 = Concatenate(axis=3)([u_prime, v_prime, w_prime])

    model = Model(inputs=x_t, outputs=x_t_plus_1, name='coupled_lorenz_transition_model')

    return model