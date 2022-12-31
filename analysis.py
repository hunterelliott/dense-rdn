"""
A module with various tools for analyzing models and their outputs/trajectories etc.
"""

import logging

import numpy as np

from layers import IterationLayerBase
from sampling import get_train_z
from ops import logits_loss, accuracy, pearson_corr


def propagate_x(x_0, transition_model, n_t, training=True, verbose=True, seed=42):
    """
    Propagates a batch of Xs through time using the input transition model.
    Args:
        x_0: 4D-tensor batch of initial state tensor for each X, (batch, height, width, channels)
        transition_model: Model which maps a batch of X_t to X_t+1
        n_t: Number of time steps to take.
        training: Whether to use training-time (vs test-time) behavior of the transition model.
        verbose: If true, log progress.

    Returns:
        x_vs_t: (n_t + 1, batch, height, width, channels) tensor of each X over time
            (includes x_0, hence the n_t + 1 dimension)
    """

    if verbose:
        logging.info("Iterating X for {} time steps...".format(n_t))

    # We do this in a simple loop - although this under-utilizes the GPUs it will allow us to use the larger CPU
    # memory to accumulate x_vs_t
    x_vs_t = np.zeros((n_t + 1,) + x_0.shape, dtype=np.float32)
    x_t = x_0
    for i_t in range(n_t + 1):
        x_vs_t[i_t, :, :, :, :] = x_t
        x_t = transition_model(x_t, i_t=i_t, seed=seed, training=training)
        if verbose and i_t % 50 == 0:
            logging.info("...at t={}".format(i_t))

    return x_vs_t


def propagate_x_and_z(model, n_t=None, d_t=None, z_bits=None, batch_size=32, use_train_time=True, seed=42):
    """
    Propagates X through time and (optionally) encodes at every timepoint using the input propagation model.
    This is done in a for loop using CPU memory so is slow but reduces GPU memory requirements.
    Args:
        model: propagation_model. If model contains an encoder z-propagation will be performed as well.
        n_t: Number of timepoints, if different from that in input model.
        d_t: Step size, if different from that in input transition model.
        z_bits: Bits of entropy in Z, if different than default.
        batch_size: Number of samples to propagate
        use_train_time: Whether to use "training=True" when calling models, which will e.g. force use of train-time
            batch normalization stats.

    Returns:
        x_vs_t: X at each timepoint, with the first axis being time.
        z_vs_t: Z at each timepoint, with the first axis being time.
        z_acc_vs_t: Accuracy of reconstructed Z at each timepoint
        z_dist_vs_t: Distance from Z_0 at each timepoint (relative entropy)

    """

    generator_model, transition_model, iteration_layer, encoder_model = get_compound_model_components(model)

    if d_t is not None:
        logging.info("Overriding model d_t of {} with d_t of {}".format(transition_model.d_t, d_t))
        transition_model.d_t = d_t

    z_shape = model.input.shape[1::]
    z_0 = get_train_z(z_shape, batch_size, entropy=z_bits).astype(np.float32)
    x_0 = generator_model(z_0, training=use_train_time)
    x_shape = x_0.shape[1::]
    if n_t is None: n_t = iteration_layer.n_t
    logging.info("Performing continuous-time iteration for {} timesteps.".format(n_t))

    # Fuck it, just loop over time. Slow, but saves memory.
    x_t = x_0
    z_dist_vs_t = np.zeros((n_t + 1, batch_size))
    z_acc_vs_t = np.zeros(n_t + 1)
    x_vs_t = np.zeros((n_t + 1, batch_size) + x_shape, dtype=np.float32)
    z_vs_t = np.zeros((n_t + 1, batch_size) + z_shape, dtype=np.float32)

    for i_t in range(n_t + 1):

        if encoder_model is not None:
            z_t = encoder_model(x_t, training=use_train_time)
            z_dist_vs_t[i_t, :] = logits_loss(z_0, z_t)
            z_acc_vs_t[i_t] = np.mean(np.array(accuracy(z_0, z_t)))
            z_vs_t[i_t, :, :, :] = z_t

        x_vs_t[i_t, :, :, :] = x_t
        x_t = transition_model(x_t, i_t=i_t, seed=seed, training=use_train_time)

        if i_t % 50 == 0:
            logging.info("at t={}".format(i_t))

    return x_vs_t, z_vs_t, z_acc_vs_t, z_dist_vs_t


def get_compound_model_components(model):
    """
    Extracts the individual generator, encoder and transition models from the input composite model,
    e.g. a Z-propagation model or a replication model.

    Args:
        model: keras Model as produced by models.iteration.propagation_model

    Returns:
        generator_model, transition_model, iteration_layer, encoder_model

    """

    generator_model = model.get_layer('state_generator_model')
    iteration_layer = [layer for layer in model.layers if isinstance(layer, IterationLayerBase)][0]
    transition_model = iteration_layer.transition_model
    try:
        encoder_model = model.get_layer('state_encoder_model')
    except:
        # Some models don'e include the encoder.
        encoder_model = None

    return generator_model, transition_model, iteration_layer, encoder_model


def pearsons_corr_vs_time(x_vs_t, target, chan_map):
    """
    Calculates the pearson's correlation for every timepoint in a channel of the input time series x_vs_t with a channel
    in the fixed traget.
    Args:
        x_vs_t: 5D [n_t, batch, height, width, channel] tensor
        target: 3D [height, width, channel] tensor to correlate with.
        chan_map: tuple, chan_map[0] of target is correlated with chan_map[1] of x_vs_t

    Returns:
        mean_r_vs_t, std_r_vs_t: n_t length vector of mean and STD of correlation values at each timepoint.
    """
    # We just loop over samples and time since we don't need speed here
    r_vs_t = []
    w, h = target.shape[0:2]
    for i_samp in range(x_vs_t.shape[1]):
        r_vs_t.append(np.array([pearson_corr(np.reshape(x[i_samp, :, :, chan_map[1]], (1, w, h, 1)),
                                             np.reshape(target[:, :, chan_map[0]], (1, w, h, 1))) for x in x_vs_t]))

    return np.mean(np.concatenate(r_vs_t, axis=-1), axis=-1), np.std(np.concatenate(r_vs_t, axis=-1), axis=-1)
