"""
Tests of Z->X "generator" mapping models
"""
import pytest
import numpy as np

from sampling import get_train_z
from models.mapping import hybrid_generator_model


@pytest.fixture
def numpy_z():
    return get_train_z((1, 1, 256), 4, entropy=32)


@pytest.fixture
def numpy_z2():
    # Lets us compare with two values, probably a better way but....
    return get_train_z((1, 1, 256), 4, entropy=32)


def test_hybrid_generator_model(numpy_x_0, numpy_z, numpy_z2):
    """
    basic tests - we confirm that only the constant areas are constant and the output is the right shape
    """

    x_pad = 8
    x_shape = (16, 16, 2)
    generator = hybrid_generator_model(x_shape, numpy_z.shape[1:], x_pad=x_pad)

    x_1 = generator(numpy_z)

    x_2 = generator(numpy_z2)

    # Input shape should match output shape
    assert x_1.shape == numpy_x_0.shape

    # The areas outside the center should be the same
    assert np.allclose(x_1[:, :, 0:x_pad - 1, :], x_2[:, :, 0:x_pad - 1, :])
    assert np.allclose(x_1[:, 0:x_pad - 1, :, :], x_2[:, 0:x_pad - 1, :, :])

    # The center should be different
    np.allclose(x_1[:, x_pad:2 * x_pad, x_pad:2 * x_pad, :], x_2[:, x_pad:2 * x_pad, x_pad:2 * x_pad:])
    np.allclose(x_1[:, x_pad:2 * x_pad, x_pad:2 * x_pad, :], x_2[:, x_pad:2 * x_pad, x_pad:2 * x_pad:])
