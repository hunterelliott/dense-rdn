"""
Runs some sanity checks on various loss functions.
"""

import pytest

import numpy as np

from ops import x_chan_mean_loss
from sampling import get_train_z


def stack_xs(x_0, x_t):
    return np.concatenate([np.expand_dims(x_0, -1),
                           np.expand_dims(x_t, -1)], axis=-1)


@pytest.fixture
def z_true():
    return get_train_z((1, 1, 256), 8)


@pytest.fixture
def x_stack_rand(numpy_x_0, numpy_x_t):
    return stack_xs(numpy_x_0, numpy_x_t)


@pytest.fixture(params=[x_chan_mean_loss])
def x_loss(request):
    return request.param


def test_x_loss_output_format(x_loss, z_true, x_stack_rand):

    loss = x_loss(z_true, x_stack_rand)

    assert loss.shape == ()


def test_x_chan_mean_loss_sanity(z_true, numpy_x_0, numpy_x_t):

    x_stack_ident = stack_xs(numpy_x_0, numpy_x_0)

    assert x_chan_mean_loss(z_true, x_stack_ident) == 0

    x_stack_diff = stack_xs(numpy_x_0, numpy_x_t)

    assert not np.allclose(x_chan_mean_loss(z_true, x_stack_diff), 0)

    # Different values but same mean should also give zero loss
    x_match_mean = numpy_x_t / np.mean(numpy_x_t, axis=(1, 2), keepdims=True) * \
                   np.mean(numpy_x_0, axis=(1, 2), keepdims=True)
    x_stack_mean_match = stack_xs(numpy_x_0, x_match_mean)

    assert not np.allclose(numpy_x_0, x_match_mean)
    assert np.allclose(x_chan_mean_loss(z_true, x_stack_mean_match), 0)

    # Same overall mean but diff channel mean should give non-zero loss
    x_stack_perm = stack_xs(numpy_x_0, np.flip(numpy_x_0, axis=3))
    assert not np.allclose(x_chan_mean_loss(z_true, x_stack_perm), 0)

    # Different spatial arrangement but same mean should give zero loss
    x_stack_perm = stack_xs(numpy_x_0, np.flip(numpy_x_0, axis=1))
    assert np.allclose(x_chan_mean_loss(z_true, x_stack_perm), 0)
