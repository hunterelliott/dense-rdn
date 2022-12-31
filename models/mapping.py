"""
This module contains models for mapping functions e.g. which map between Z<->X
"""
import os
import math
import numpy as np
import logging

from tensorflow.keras.layers import Conv2D, Input, Conv2DTranspose, ZeroPadding2D, Cropping2D, Lambda
from tensorflow.keras.layers import BatchNormalization, Multiply, GaussianNoise, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.initializers import Constant
import tensorflow as tf

from utils import load_model
from analysis import get_compound_model_components
from layers import GaussianFilterLayer, RescalingLayer, ChannelOffsetLayer, ClippingLayer


def encoder_model(x_shape, z_shape, model_name, **kwargs):
    """
    Wrapper for retrieving an encoder model by name.
    Args:
        x_shape: shape of state space
        z_shape: shape of representation space Z
        model_name: name of function in this module which will return the model. OR a directory of a composite
            (e.g. z-propagation) model to load the encoder from.
        **kwargs: arguments to pass to the model constructor function.

    Returns:
        model: a keras Model object.

    """
    if os.path.exists(model_name):
        logging.info("Loading encoder:")
        model = get_compound_model_components(load_model(model_name))[3]
        assert model.input_shape[1::] == x_shape and model.output_shape[1::] == z_shape, \
            "The loaded model does not match the expected input and output shape!"
    else:
        assert 'encoder' in model_name, "If a name is input it must be of an encoder!"
        logging.info("Building encoder {}".format(model_name))
        model = globals()[model_name](x_shape, z_shape, **kwargs)
    return model


def simple_encoder_model(x_shape, z_shape, x_pad=None, name='state_encoder_model'):
    """
    Defines a model which maps from the state space X to the representation space Z
    Args:
        x_shape: shape of state space *after* input padding/cropping.
        z_shape: shape of representation space Z
        x_pad: (optional) pass this to Cropping2D to zero-pad the input (i.e. crop) of the encoder.
        name: (optional) define a custom name

    Returns:
        model: a Keras Model object.

    """

    assert x_shape[0] == x_shape[1], "This model assumes a square domain"
    assert z_shape[0] == z_shape[1], "This model assumes a square Z"


    stride = 2
    kernel_size = 4
    n_layers = int(math.log(x_shape[0], stride) - math.log(z_shape[0], stride))
    layer_width = np.linspace(x_shape[2], z_shape[2], num=n_layers).astype(np.int64)

    x = Input(shape=x_shape)

    if x_pad is not None:
        h = Cropping2D(x_pad)(x)
    else:
        h = x

    for i_layer in range(n_layers):
        if i_layer == n_layers - 1:
            # We output logits rather than probabilities
            activation = None
        else:
            activation = 'tanh'

        h = Conv2D(layer_width[i_layer], kernel_size, strides=(stride, stride), padding='same',
                   activation=activation)(h)

        if activation:
            h = BatchNormalization(epsilon=1e-1)(h, training=True)

    # Shift output to match Z range (within models we are zero-centered and in [-1,1])
    # While not requisite this improves convergence rate.
    de_centering_layer = Lambda(lambda x: (x / 2.0 + 0.5))
    z = de_centering_layer(h)

    return Model(inputs=x, outputs=z, name=name)


def mobilenetv2_pretrained_encoder_model(x_shape, z_shape, x_pad=None, name='state_encoder_model'):
    return mobilenetv2_encoder_model(x_shape, z_shape, x_pad=x_pad, name=name, weights='imagenet')


def mobilenetv2_encoder_model(x_shape, z_shape, x_pad=None, name='state_encoder_model', weights=None):
    """
        Defines a model which maps from the state space X to the representation space Z and which uses MobileNetV2 as
        it's core "feature extractor".
        Args:
            x_shape: shape of state space
            z_shape: shape of representation space Z
            x_pad: (optional) zero-pad the input (i.e. crop) of the encoder by this amount.
            name: (optional) define a custom name
            weights: (optional) either None for random initialization or 'imagenet' for a pre-trained model.

        Returns:
            model: a Keras Model object.

    """
    if weights is not None:
        logging.info("Using weights from {}".format(weights))

    x = Input(shape=x_shape)

    if x_pad is not None > 0:
        h = Cropping2D(x_pad)(x)
    else:
        h = x

    if x_shape[0] < 96:
        # Depth of this model requires an input of a minimum size
        h = tf.image.resize(h, (96, 96))

    h = MobileNetV2(input_shape=h.shape[1::], classes=z_shape[-1], weights=weights, include_top=False)(h)
    # Replace the top to match our output (required to allow use of imagenet weights)
    z = Conv2D(z_shape[-1], h.shape[1:3])(h)

    return Model(inputs=x, outputs=z, name=name)


def complement_encoder_model(x_shape, z_shape, x_pad, base_encoder_model='simple_encoder_model',
                             name='state_encoder_model'):
    """
    An encoder model which sees the complement of the sub-domain of X which is seen by the standard encoder model when
    x_pad > 0. This is achieved by masking out the center portion which would have been fed into the normal
    encoder after cropping.

    Args:
        x_shape: same as normal encoder
        z_shape: same as normal encoder
        x_pad: same as normal encoder
        base_encoder_model: the encoder model to pass the complemented (masked) X into.
        name: optionally specify a custom name for the model.

    Returns:
        model = a keras Model object.

    """
    x = Input(shape=x_shape)

    assert len(x_pad) == 1, "This model expects scalar (symmetric) padding!"

    # Mask out the central portion which is seen by the standard encoder
    mask = tf.pad(tf.zeros_like(Cropping2D(x_pad)(x)),
                  [[0, 0], [x_pad, x_pad], [x_pad, x_pad], [0, 0]], constant_values=1.0)
    x_masked = Multiply()([x, mask])

    # Call the normal encoder on the masked input, without any padding.
    encoder = encoder_model(x_shape, z_shape, base_encoder_model, name='base_encoder')
    z = encoder(x_masked)

    return Model(inputs=x, outputs=z, name=name)


def generator_model(x_shape, z_shape, model_name, **kwargs):
    """
    Wrapper for retrieving a generator model by name.
    Args:
        x_shape: shape of state space
        z_shape: shape of representation space Z
        model_name: name of function in this module which will return the model. OR a directory of a composite
            (e.g. z-propagation) model to load the encoder from.
        **kwargs: arguments to pass to the model constructor function.

    Returns:
        model: a keras Model object.

    """
    if os.path.exists(model_name):
        logging.info("Loading generator:")
        model = get_compound_model_components(load_model(model_name))[0]
    else:
        assert 'generator' in model_name, "If a name is input it must be of a generator!"
        logging.info("Building generator {}".format(model_name))
        model = globals()[model_name](x_shape, z_shape, **kwargs)
    return model


def simple_generator_model(x_shape, z_shape, x_pad=None, batch_norm=True, output_rescale=0.5,
                           name='state_generator_model'):
    """
    A model which maps from Z space to X state space.
    Args:
        x_shape: (HxWxC) shape of generated state space (does not include batch axis).
        z_shape: shape of representation space Z
        x_pad: (optional) input to pass to ZeroPadding2D for padding generator output (output will no longer match x_shape)
        batch_norm: If true, batch norm is applied after each conv layer.
        output_rescale: Initial (learned) rescaling of the output, <1 improves stability with some transition models.
        name: (optional) define a custom name

    Returns:
        model: a Keras Model object.

    """
    assert x_shape[0] == x_shape[1], "This model assumes a square X"
    assert z_shape[0] == z_shape[1], "This model assumes a square Z"

    stride = 2
    kernel_size = 4
    assert math.log(x_shape[0], stride) % 1 == 0, "generator output width must be a power of 2"
    n_layers = int(math.log(x_shape[0], stride) - math.log(z_shape[0], stride))
    layer_width = np.linspace(z_shape[2], x_shape[2], num=n_layers).astype(np.int64)

    z = Input(shape=z_shape)

    # Directly correct for difference in Z range [0,1] and model/state range [-1, 1].
    # Could be learned but this improves convergence rate.
    centering_layer = Lambda(lambda x: (x - .5) * 2)
    h = centering_layer(z)

    for i_layer in range(n_layers):
        h = Conv2DTranspose(layer_width[i_layer], kernel_size, strides=(stride, stride), padding='same')(h)

        if i_layer == n_layers-1:
            # Initialize the output to use less than the full (-1,1) range to improve optimization stability
            if batch_norm:
                h = BatchNormalization(epsilon=1e-1, gamma_initializer=Constant(value=output_rescale))(h, training=True)
            else:
                h = RescalingLayer(init_scale=output_rescale)(h)
        elif batch_norm:
            h = BatchNormalization(epsilon=1e-1)(h, training=True)

        h = Activation('tanh')(h)

        if i_layer == n_layers-1:
            # We maintain the minimum bound of -1 but allow the upper limit to be rescaled
            h = RescalingLayer(init_scale=1.0)(h + 1) - 1.0

    if x_pad is not None:
        x = ZeroPadding2D(x_pad)(h)
    else:
        x = h

    return Model(inputs=z, outputs=x, name=name)


def noised_generator_model(x_shape, z_shape, x_pad=0, base_generator_model='simple_generator_model',
                           noise_layer=GaussianNoise(1.0), name='state_generator_model', **kwargs):
    """
    A generator model for which the output is the sum of the generator output and random noise.
    Args:
        x_shape: same as normal generator.
        z_shape: same as normal generator.
        x_pad: same as normal generator.
        base_generator_model: which generator model to noise the output of.
        noise_layer: a layer to use to noise the generator output.
        name: same as normal generator.

    Returns:
        model: keras Model object.

    """

    z = Input(shape=z_shape)
    generator = generator_model(x_shape, z_shape, base_generator_model, x_pad=x_pad, name='base_generator', **kwargs)
    x_0 = generator(z)
    x_noised = noise_layer(x_0, training=True)

    return Model(inputs=z, outputs=x_noised, name=name)


def rd_generator_model(x_shape, z_shape, x_pad=None, base_generator_model='simple_generator_model',
                       fit_background_conc=True, name='state_generator_model', **kwargs):
    """
    A generator model wrapper for reaction-diffusion models which slightly modifies the specified generator to ensure
    spatially continuous output and uses zero-concentration padding.

    Args:
        x_shape: same as normal generator.
        z_shape: same as normal generator.
        x_pad: same as normal generator.
        base_generator_model: which generator model to modify for use in reaction-diffusion models.
        fit_background_conc: If true, a fittable constant offset is added to the output if padding enabled
        name: same as normal generator.

    Returns:
        model: keras Model object.

    """

    z = Input(shape=z_shape)
    # Get the base generator output without padding
    generator = generator_model(x_shape, z_shape, base_generator_model, name='base_generator', **kwargs)
    h = generator(z)

    if x_pad is not None:
        # We pad with -1, the lower bound of our domain and therefore 0 concentration within the RD model
        h = h + 1
        x = ZeroPadding2D(x_pad)(h)
        x = x - 1
    else:
        x = h

    if fit_background_conc and x_pad is not None:
        # Optionally allow a constant background concentration
        x = ChannelOffsetLayer(i_chan=np.arange(0, x_shape[-1]), init_offset=1e-4, min_value=0.0, max_value=10.0)(x)

    # And we avoid introducing discontinuities / singularities by gaussian filtering the output.
    x = GaussianFilterLayer(1.0, 3)(x)

    return Model(inputs=z, outputs=x, name=name)


def hybrid_generator_model(x_shape, z_shape, x_pad=None, base_generator_model='rd_generator_model',
                           name='state_generator_model', **kwargs):
    """
    A generator model for which the output is the sum of the generator output and random noise.
    Args:
        x_shape: same as normal generator.
        z_shape: same as normal generator.
        x_pad: same as normal generator.
        base_generator_model: which generator model to combine outputs from to make the hybrid generator.
        name: same as normal generator.

    Returns:
        model: keras Model object.

    """
    assert x_pad is not None and x_pad > 0, "This generator requires padding x_pad > 0!"

    z = Input(shape=z_shape)

    # The central generator is padded and takes in the Z vector input
    z_generator = generator_model(x_shape, z_shape, base_generator_model, x_pad=x_pad,
                                  name='base_generator_z', **kwargs)
    x_z_0 = z_generator(z)

    # The surrounding area and central background is generated with constant input (hence no batch norm)
    constant_generator = generator_model(z_generator.output_shape[1:], z_shape, base_generator_model,
                                         name='base_generator_constant', batch_norm=False, **kwargs)
    x_c_0 = constant_generator(tf.ones_like(z) * .5)

    # The final output is the sum, but we preserve the finite (-1, 1) range
    x_0 = ClippingLayer()(x_z_0 + x_c_0)

    return Model(inputs=z, outputs=x_0, name=name)