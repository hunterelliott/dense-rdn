"""
This module contains composite models which are used to solve for particular initial, final or persistent states.
"""

import math

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, Cropping2D

from models.mapping import generator_model, encoder_model
from models.transition import transition_model
import layers
import domains


def propagation_model(x_shape, z_shape, n_t,
                      x_output_period=None, iteration_layer='IterationLayer',
                      state_generator=None, state_transition_model=None, state_encoder=None,
                      x_pad_generator=None, x_pad_encoder=None, batch_size=None, name='propagation_model',
                      freeze_encoder=False, freeze_generator=False, x_outputs=(0, -1), padded_x_out=True,
                      z_outputs=True):
    """
    A model which generates X_0 from Z and then propagates it through time, possibly reconstructing Z as well.

    Args:
        x_shape: shape of state space
        z_shape: shape of representation space Z
        n_t: number of timesteps to iterate X in total
        x_output_period: how often, in time, X is output from the IterationLayer
        iteration_layer: which sub-class of IterationLayerBase to use to iteratively update X to propagate time.
        state_transition_model: model which maps X_t to X_t+1. Name, filename or Model Object.
        state_generator: Optionally input a state generator model which generates X_0 from Z. Name, filename or Model Object.
        state_encoder: Optionally input a state generator model which encodes Z_t from X_t. Name, filename or Model Object.
        x_pad_... : Optionally pad the input / output of the encoder / decoder to allow X to be larger than these models.
        name: (optional) model name for the created z propagation model.
        x_outputs: If not None, include these indices of iteration layer outputs of the X-(sub)domain for use in
            similarity losses. If the x_pad_generator is not None then this output will be cropped to match the
            generator's sub-domain.
        padded_x_out If True and x_pad_generator not None then the model outputs will be padded (both at t=0 and t=n_t)
        z_outputs: If true, include outputs of the Z-encoding of X-(sub) domain for use in Z-reconstruction losses.

    Returns:
        model: a keras Model object.

    """

    if x_output_period is not None:
        assert n_t % x_output_period == 0, "n_t must be an integer multiple of encoding period."
    else:
        x_output_period = 1

    if isinstance(state_generator, str):
        if x_pad_generator is not None:
            x_shape_gen = Cropping2D(x_pad_generator)(Input(shape=x_shape)).shape[1:]
        state_generator = generator_model(x_shape_gen, z_shape, state_generator, x_pad=x_pad_generator)

    if isinstance(state_transition_model, str):
        state_transition_model = transition_model(x_shape, state_transition_model)

    if isinstance(iteration_layer, str):
        state_iterator = getattr(layers, iteration_layer)(state_transition_model, n_t, x_output_period=x_output_period)
    else:
        state_iterator = iteration_layer

    if isinstance(state_encoder, str):
        state_encoder = encoder_model(x_shape, z_shape, state_encoder, x_pad=x_pad_encoder)

    z_0 = Input(z_shape, batch_size=batch_size)
    x_0 = state_generator(z_0)
    x_vs_t = state_iterator(x_0)

    if freeze_encoder:
        state_encoder.trainable = False
    if freeze_generator:
        state_generator.trainable = False

    outputs = []
    if z_outputs:
        if 'stochastic' in state_iterator.name:
            z_vs_t = []
            for i_out in range(len(state_iterator.output)):
                z_vs_t.append(state_encoder(x_vs_t[i_out]))
        else:
            z_vs_t = []
            for t in range(0, int(n_t / state_iterator.x_output_period) + 1, 1):
                z_vs_t.append(state_encoder(x_vs_t[t]))

        outputs = z_vs_t

    if x_outputs:
        if x_pad_generator is not None and not padded_x_out:
            x_0_hat = Cropping2D(x_pad_generator)(x_vs_t[x_outputs[0]])
            x_t_hat = Cropping2D(x_pad_generator)(x_vs_t[x_outputs[1]])
        else:
            x_0_hat = x_vs_t[x_outputs[0]]
            x_t_hat = x_vs_t[x_outputs[1]]
        # Do this hacky shit to follow keras loss/output specs
        x_stack_t = Concatenate(axis=-1, name='x_output')([tf.expand_dims(x_0_hat, axis=-1),
                                                           tf.expand_dims(x_t_hat, axis=-1)])
        outputs = outputs + [x_stack_t]

    return Model(inputs=z_0, outputs=outputs, name=name)


def replication_model(x_shape, z_shape, n_t,
                      iteration_layer='IterationLayer',
                      state_generator=None, state_transition_model=None, state_encoder=None,
                      x_buffer=0, x_yard=0, batch_size=None, x_outputs=True, z_outputs=True,
                      generator_kwargs={}, encoder_kwargs={}, transition_kwargs={}, name='replication_model',
                      granddaughter_fraction=0, n_t_x_loss=1, x_0_preproc_layer=None):
    """
    A model which "replicates" a generated "parent" X by embedding this X as a sub-domain in a larger domain (X_full),
    propagating it through time, and then extracting two sub-domains corresponding to "daughter" replicates. These
    daughters may also have an encoding of the Z vector which was used to generate X, as a form of "heritable"
    information and variation.

    Args:
        x_shape: shape of the parent sub-domain to be generated.
        z_shape: shape of the Z vector used to generate X
        n_t: number of time points to allow for replication.
        iteration_layer: layer to use to encapsulate transition model.
        state_generator: model which generates X from Z
        state_transition_model: model which propagates X through time.
        state_encoder: model which maps an X sub-domain back to a Z vector.
        x_buffer: width of buffer background region to include around each X.
        x_yard: Additional spacing outside of daughters for subsequent generations.
        batch_size: minibatch size.
        x_outputs: If true, include outputs of the X-subdomain corresponding to each daughter (will include buffer)
        z_outputs: If true, include outputs of the Z-encoding of each daughter (encoder input does not include buffer)
        generator_kwargs: Additional arguments to pass to generator model constructor.
        encoder_kwargs: Additional arguments to pass to encoder model constructor.
        transition_kwargs: Additional arguments to pass to transition model constructor.
        name: Optionally specify a different model name.
        granddaughter_fraction: What fraction of each batch to pass through a second round of replication before
            applying loss.
        n_t_x_loss: Number of timepoints over which to compare Xs when applying X-similarity losses.
        x_0_preproc_layer: Optionally include a layer to pre-process (e.g. noise) the X0 state before propagating.

    Returns:
        mode: A keras Model object.

    """
    assert x_outputs or z_outputs, "At least one of x_outputs or z_outputs must be True."

    # Create the domain geometry
    rep_domain = domains.ReplicationDomain(x_shape, buffer=x_buffer, yard=x_yard)

    # We pad the generator output to create the full domain for propagation through time
    x_pad_generator = domains.get_domain_padding(rep_domain.full_domain, rep_domain.parent_domain)

    if isinstance(state_generator, str):
        state_generator = generator_model(x_shape, z_shape, state_generator, x_pad=x_pad_generator, **generator_kwargs)

    if isinstance(state_encoder, str):
        state_encoder = encoder_model(x_shape, z_shape, state_encoder, **encoder_kwargs)

    z_0 = Input(z_shape, batch_size=batch_size)
    x_full_0_raw = state_generator(z_0)
    if x_0_preproc_layer is not None:
        # We have a separate variable for the pre-processed (e.g. noised) X0 so the raw X0 can be used in losses.
        x_full_0 = x_0_preproc_layer(x_full_0_raw)
    else:
        x_full_0 = x_full_0_raw

    if isinstance(state_transition_model, str):
        # We have to delay creation of the transition model until after the full domain size is specified.
        state_transition_model = transition_model(state_transition_model, **transition_kwargs)
    state_iterator = getattr(layers, iteration_layer)(state_transition_model, n_t, x_output_period=n_t)

    x_full_t = state_iterator(x_full_0)[-1]

    if granddaughter_fraction > 0:
        if granddaughter_fraction < 1:
            # Split off a subset of the batch to form a second generation
            n_gd = tf.cast(tf.math.ceil(x_full_0.shape[0] * granddaughter_fraction), dtype=tf.int32)
            [x_d, x_gd] = tf.split(x_full_t, (x_full_0.shape[0]-n_gd, n_gd))
        else:
            x_gd = x_full_t

        # Randomly center either the left or right sub-batch of daughters
        daughter_offset = int(rep_domain.parent_domain.center[1] - rep_domain.daughter_domains[1].center[1])
        x_gd = layers.RandomDaughterCenteringLayer(daughter_offset)(x_gd)
        if x_0_preproc_layer is not None:
            # preprocess t=0 for the second generation as well.
            x_gd = x_0_preproc_layer(x_gd)
        # ...and then propagate them through time to form granddaughters before recombining
        x_gd = state_iterator(x_gd)[-1]

        if granddaughter_fraction < 1:
            x_full_t = tf.concat([x_d, x_gd], axis=0)
        else:
            x_full_t = x_gd

    outputs = []
    if z_outputs:
        # The parent and two daughter X sub-domains, cropped from x_full_t, and their encoding.
        x_0, x_1_t, x_2_t = layers.ReplicationOutputLayer(rep_domain, include_buffer=False)(x_full_0, x_full_t)
        z_0_prime = state_encoder(x_0)  # We re-encode X_0 since our generator is not trivially invertible.
        z_1_t = state_encoder(x_1_t)
        z_2_t = state_encoder(x_2_t)

        outputs = [z_0_prime, z_1_t, z_2_t]

    if x_outputs:
        if n_t_x_loss > 1:
            xvt_iterator = getattr(layers, iteration_layer)(state_transition_model, n_t_x_loss, x_output_period=1)
            # The X-similarity losses use the raw un-noised/un-processed X0 to induce robustness
            x_full_0_raw = xvt_iterator(x_full_0_raw)
            x_full_t = xvt_iterator(x_full_t)
        else:
            x_full_0_raw = tf.expand_dims(x_full_0_raw, axis=0)
            x_full_t = tf.expand_dims(x_full_t, axis=0)

        # To avoid periodic/tiling solutions the x-similarity loss includes the buffer
        x_hat_0, x_1_hat_t, x_2_hat_t = layers.ReplicationOutputLayer(rep_domain, include_buffer=True)(x_full_0_raw, x_full_t)

        # Do this hacky shit to follow keras loss/output specs
        x_stack_t = Concatenate(axis=-1, name='x_output')([tf.expand_dims(x_hat_0, axis=-1),
                                                           tf.expand_dims(x_1_hat_t, axis=-1),
                                                           tf.expand_dims(x_2_hat_t, axis=-1)])

        outputs = outputs + [x_stack_t]

    return Model(inputs=z_0, outputs=outputs, name=name)
