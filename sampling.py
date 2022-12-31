"""
This module contains functions for sampling from distributions e.g. for optimization
"""

import numpy as np


def get_train_z(z_shape, batch_size, entropy=None):
    """
    Returns a batch of binary Z vectors for use in training.
    Args:
        z_shape: dimensions of Z, should be 3-tuple
        batch_size: number of distinct Z vectors. Will be 1st dimension
        entropy: bits of entropy in returned Z. If not specified it will be maximal Z dimension

    Returns:
        Z of shape (batch_size,) + z_shape
    """

    z = np.random.choice([0.0, 1.0], size=(batch_size,) + z_shape)

    if entropy is not None and entropy < z_shape[-1]:
        # Optionally return a Z with specified entropy
        z[:, :, :, entropy:] = 0.0

    return z


def generate_train_z(z_shape, batch_size, **kwargs):
    """
    Generator for training with models that input and then reconstruct Z
    Args:
        z_shape: dimensions of Z, should be 3-tuple
        batch_size: number of distinct Z vectors per yield. Will be 1st dimension.
        **kwargs: all additional arguments passed on to get_train_Z

    Returns:
        tuples of identical batches of (Z, Z)

    """
    while True:
        z = get_train_z(z_shape, batch_size, **kwargs)
        yield (z, z)


def generate_train_z_multi_output(z_shape, batch_size, n_outputs, **kwargs):
    """
    Generator for training models which reconstruct z multiple times.
    Args:
        z_shape: dimensions of Z, should be 3-tuple
        batch_size: number of distinct Z vectors per yield. Will be 1st dimension.
        n_outputs: number of outputs which reconstruct z
        **kwargs: all additional arguments passed on to get_train_Z

    Returns:
        tuples of identical (Z, [Z_0, ... Z_n_outputs])

    """

    while True:
        z = get_train_z(z_shape, batch_size, **kwargs)
        yield (z, [z for _ in range (n_outputs)])
