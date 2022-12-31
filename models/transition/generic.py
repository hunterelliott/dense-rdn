"""
Some generic non-physically-motivated transition models.
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.initializers.initializers_v2 import Constant, TruncatedNormal
from tensorflow.python.keras.layers import Lambda, GaussianNoise, Conv2D, Conv2DTranspose, BatchNormalization

from layers import DisplacementLayer
from ops import pad_toroidal, channel_averaging_kernel


def lambda_transition_model(x_shape, transition_function=Lambda(lambda x: GaussianNoise(10.0)(x, training=True)),
                            name='lambda_transition_model'):
    """
    Simple transition model for arbitrary lambda function transformations of X

    Args:
        x_shape: 3-tuple specifying shape of X domain (exlcudes batch dimension)
        transition_function: A keras Lambda function, with output same shape as input
        name: name for the model.

    Returns:
        model: a keras Model object.

    """
    x_t = Input(shape=x_shape)
    x_t_plus_1 = transition_function(x_t)

    return Model(inputs=x_t, outputs=x_t_plus_1, name=name)


def convolutional_transition_model(x_shape, kernel_size=3):
    """
    A simple learnable convolutional transition model.
    Args:
        x_shape: shape of state to propagate.
        kernel_size: side of kernes in convolution and conv transpose.

    Returns:
        model: a keras Model object

    """

    x_t = Input(shape=x_shape)
    assert kernel_size % 2 == 1, "We expect an odd kernel size!"
    x_t_plus_1 = pad_toroidal(x_t, int((kernel_size-1)/2))
    x_t_plus_1 = Conv2D(x_shape[-1], kernel_size=kernel_size, padding='valid', activation='tanh',
                        kernel_initializer=Constant(1 / (kernel_size**2 * x_shape[-1]) * 1.5),
                        bias_initializer='zeros')(x_t_plus_1)

    return Model(inputs=x_t, outputs=x_t_plus_1, name='convolutional_transition_model')


def convolutional_hidden_layer_transition_model(x_shape, stride=2, kernel_size=3, dimensionality_scaling=2):
    """
    A simple learnable convolutional transition model with a single hidden layer, possibly of lower spatial resolution.
    Args:
        x_shape: shape of state to propagate.
        stride: stride of convolutions and conv transpose kernels.
        kernel_size: side of kernes in convolution and conv transpose.
        dimensionality_scaling: overall dimensionality of the hidden layer relative to the input X state.

    Returns:
        model: a keras Model object

    """

    x_t = Input(shape=x_shape)

    h = Conv2D(x_shape[-1] * dimensionality_scaling * stride**2, kernel_size=kernel_size, strides=(stride, stride),
               padding='same', activation='tanh')(x_t)

    if stride > 1:
        x_t_plus_1 = Conv2DTranspose(x_shape[-1], kernel_size=kernel_size, strides=(stride, stride),
                                     padding='same', activation='tanh')(h)
    else:
        x_t_plus_1 = Conv2D(x_shape[-1], kernel_size=kernel_size, strides=(stride, stride), padding='same',
                            activation='tanh')(h)

    return Model(inputs=x_t, outputs=x_t_plus_1, name='convolutional_hidden_layer_transition_model')


def local_mlp_transition_model(x_shape, kernel_size=3, n_layers=3, dimensionality_scaling=4, batch_norm=False,
                               n_chan_average=None, differential=False):
    """
    A trainable transition model where the mapping for each element is a multilayer perceptron, the input of which is
    that element's neighborhood.

    Args:
        x_shape: size of X (state)
        kernel_size: kernel size - determines how large a neighborhood the MLP takes as input.
        n_layers: number of layers.
        dimensionality_scaling: number of neurons in first layer relative to each state element.
        batch_norm: If True, apply batch norm at each layer.
        n_chan_average: If none, then the input to the MLP is a learned convolution. If >0 then the input to the MLP is
            the raw local state vector with this number of channels locally averaged, and the rest passed unaltered.

    Returns:
        model: A keras Model object.

    """

    layer_width = np.linspace(x_shape[-1] * dimensionality_scaling, x_shape[-1], n_layers, dtype=np.int)

    x_t = Input(shape=x_shape)
    assert kernel_size % 2 == 1, "We expect an odd kernel size!"
    h = pad_toroidal(x_t, int((kernel_size-1)/2))

    if n_chan_average is None:
        h = Conv2D(layer_width[0], kernel_size=kernel_size, padding='valid', activation='tanh')(h)
    else:
        kernel = channel_averaging_kernel((kernel_size, kernel_size, x_shape[-1], 1), tf.float32,
                                          n_chan_average=n_chan_average)
        h = tf.nn.depthwise_conv2d(h, kernel, (1, 1, 1, 1), "VALID")

    for i_layer in range(1, n_layers):

        h = Conv2D(layer_width[i_layer], kernel_size=1, padding='valid', activation='tanh')(h)
        if batch_norm:
            h = BatchNormalization(epsilon=1e-1)(h)
    if differential:
        x_t_plus_1 = x_t + h
    else:
        x_t_plus_1 = h

    return Model(inputs=x_t, outputs=x_t_plus_1, name='local_mlp_transition_model')


def generic_nonlinear_local_learnable_transition_model(x_shape, epsilon=1e-3, kernel_size=3):
    """
    A model which allows learning a transition function from the space of all "locally coupled" first-order nonlinear
    differential dynamical systems. Locally coupled here is defined as each derivative being a (possibly nonlinear)
    function of the average of its neighbor's values.

    Args:
        x_shape: shape of state to propagate.
        epsilon: integration step size

    Returns:
        model: a keras Model object.

    """

    x_t = Input(shape=x_shape)

    x_t_plus_1 = GenericNonlinearLocalLearnableTransitionLayer(x_shape, epsilon=epsilon, kernel_size=kernel_size)(x_t)

    return Model(inputs=x_t, outputs=x_t_plus_1, name='generic_nonlinear_local_learnable_transition_model')


class GenericNonlinearLocalLearnableTransitionLayer(Layer):
    """
    See similarly-named method above - we had to encapsulate the actual machinery in a layer to allow weights to be
    trainable.
    """
    def __init__(self, x_shape, epsilon=1e-1, kernel_size=3):
        super(GenericNonlinearLocalLearnableTransitionLayer, self).__init__()

        self.x_shape = x_shape
        self.epsilon = epsilon
        assert kernel_size % 2 == 1, "We expect an odd kernel size!"
        self.kernel_size = kernel_size
        self.a = self.add_weight(shape=(x_shape[-1], x_shape[-1]), initializer=TruncatedNormal(10.0), trainable=True, name="a")
        self.b = self.add_weight(shape=(x_shape[-1], x_shape[-1]), initializer=TruncatedNormal(10.0), trainable=True, name="b")
        self.c = self.add_weight(shape=(x_shape[-1], x_shape[-1]), initializer=TruncatedNormal(10.0), trainable=True, name="c")
        self.bias = self.add_weight(shape=(3,), initializer='zeros', trainable=True, name="bias")

        initializer = lambda shape, dtype: channel_averaging_kernel(shape, dtype, n_chan_average=3)
        self.w = self.add_weight(shape=(kernel_size, kernel_size, x_shape[-1], 1),
                                 initializer=initializer, trainable=False, name="w")

    def call(self, x_t):

        k = self.kernel_size

        # TODO - add linear terms? remove redundant upper diagonal nonlinear terms?
        # First perform a local average of each element of the local state vector, with toroidal boundary conditions.
        x_t_hat = pad_toroidal(x_t, int((k - 1) / 2))
        x_t_hat = tf.nn.depthwise_conv2d(x_t_hat, self.w, (1, 1, 1, 1), "VALID")

        # Outer product of the state vector at each position in x (x[i, j, :]), allowing for all possible first-order
        # nonlinearities
        outer_u = tf.einsum('abci,abcj->abcij', x_t_hat, x_t_hat)
        outer_v = tf.einsum('abci,abcj->abcij', x_t_hat, x_t_hat)
        outer_w = tf.einsum('abci,abcj->abcij', x_t_hat, x_t_hat)

        # Each element of the outer product has a trainable weight before summation to give it's derivative
        delta_u = tf.einsum('abcij,ij->abc', outer_u, self.a) + self.bias[0]
        delta_v = tf.einsum('abcij,ij->abc', outer_v, self.b) + self.bias[1]
        delta_w = tf.einsum('abcij,ij->abc', outer_w, self.c) + self.bias[2]

        # Integrate
        u_t_plus_1 = x_t[:, :, :, 0] + self.epsilon * delta_u
        v_t_plus_1 = x_t[:, :, :, 1] + self.epsilon * delta_v
        w_t_plus_1 = x_t[:, :, :, 2] + self.epsilon * delta_w

        x_t_plus_1 = tf.stack([u_t_plus_1, v_t_plus_1, w_t_plus_1], axis=-1)

        return x_t_plus_1

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(GenericNonlinearLocalLearnableTransitionLayer, self).get_config()
        config.update({"x_shape": self.x_shape,
                       "epsilon": self.epsilon,
                       "kernel_size": self.kernel_size,
                       # "a": self.a,
                       # "b": self.b,
                       # "c": self.c,
                       })
        return config


def convolutional_displacement_transition_model(x_shape, kernel_size=3, n_layers=1, dimensionality_scaling=2,
                                                maximum_displacement=2**0.5, batch_size=None):

    assert batch_size is not None, "This model requires pre-specified batch size!"
    x_t = Input(shape=x_shape, batch_size=batch_size)
    assert n_layers % 2 == 1, "We require an odd number of hidden layers!"
    assert kernel_size % 2 == 1, "We expect an odd kernel size!"

    layer_widths = [x_shape[-1]*dimensionality_scaling*(i+1) for i in range(int((n_layers+1)/2))]
    # The dimensionality progression is symmetric about the waist, except the last layer which emits a tuple of
    # (i,j) displacements for each element
    layer_widths = layer_widths + layer_widths[-2::-1] + [2]

    h = x_t
    for layer_width in layer_widths:
        h = pad_toroidal(h, int((kernel_size - 1) / 2))
        h = Conv2D(layer_width, kernel_size=kernel_size, padding='valid', activation='tanh')(h)
    h = h * maximum_displacement

    # The last layer gives velocity vectors which we use to displace the previous timepoint.
    x_t_plus_1 = DisplacementLayer()(x_t, h)

    return Model(inputs=x_t, outputs=x_t_plus_1, name='convolutional_displacement_transition_model')