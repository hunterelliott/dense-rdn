import pytest
import numpy as np

from tensorflow.keras.layers import Lambda
import tensorflow as tf

import analysis


@pytest.fixture(params=[{'n_t': 4, 'x_output_period': 1},
                        {'n_t': 8, 'x_output_period': 2},
                        {'n_t': 6, 'x_output_period': 3},
                        {'n_t': 4, 'x_output_period': 4}])
def n_t_params(request):
    return request.param


def simple_transition_model(x, **kwargs):
    return x + 1


def i_t_input_test_transition_model(x, i_t=0, **kwargs):
    return tf.ones_like(x) * tf.cast(i_t, tf.float32)


def test_standard_iterator_output_structure(standard_iterator_class, n_t_params):

    iterator = standard_iterator_class(simple_transition_model,
                                       n_t=n_t_params['n_t'],
                                       x_output_period=n_t_params['x_output_period'])

    x_0 = np.zeros((4, 8, 8, 3))
    x_vs_t = iterator(x_0)

    # Make sure we output the correct number of timepoints
    assert x_vs_t.shape[0] == n_t_params['n_t'] / n_t_params['x_output_period'] + 1

    # Check that each timepoint is the right shape
    assert x_vs_t[-1].shape == x_0.shape

    # And x_0 should be included in output as first element
    assert np.all(x_vs_t[0] == x_0)

    # And a sanity check that the correct number of iterations was run
    assert x_vs_t[-1][0, 0, 0, 0] == n_t_params['n_t']


def test_standard_iterator_i_t_inputs(standard_iterator_class, n_t_params):

    iterator = standard_iterator_class(i_t_input_test_transition_model,
                                       n_t=n_t_params['n_t'],
                                       x_output_period=n_t_params['x_output_period'])

    x_0 = np.zeros((4, 8, 8, 3))
    x_vs_t = iterator(x_0)

    # Using this test iterator the state should reflect the i_t input
    assert x_vs_t[0][0, 0, 0, 0] == 0
    assert x_vs_t[1][0, 0, 0, 0] == n_t_params['x_output_period'] - 1
    assert x_vs_t[-1][0, 0, 0, 0] == (n_t_params['n_t'] - 1)


def test_standard_iterator_vs_manual(standard_iterator, numpy_x_0):

    # First "manually" iterate the state
    x_t_manual = numpy_x_0
    seed = standard_iterator.rng.uniform_full_int((), dtype=tf.int32)
    for i_t in range(standard_iterator.n_t):
        x_t_manual = standard_iterator.transition_model(x_t_manual, i_t=i_t, seed=seed)
    standard_iterator.rng.reset_from_seed(standard_iterator.seed)

    # Now run through iterator and compare
    x_t_iter = standard_iterator(numpy_x_0)

    assert np.allclose(numpy_x_0, x_t_iter[0])
    assert np.allclose(x_t_manual, x_t_iter[-1])

    assert x_t_iter.shape[0] == int(standard_iterator.n_t / standard_iterator.x_output_period) + 1


def test_standard_iterator_vs_propagate(standard_iterator, numpy_x_0):

    # Use the iteration wrapper to output a dense timeseries
    seed = standard_iterator.rng.uniform_full_int((), dtype=tf.int32)
    x_vs_t_prop = analysis.propagate_x(numpy_x_0, standard_iterator.transition_model, standard_iterator.n_t, seed=seed)
    standard_iterator.rng.reset_from_seed(standard_iterator.seed)

    # Force the iterator to be dense so we can compare each timepoint
    standard_iterator.x_output_period = 1
    x_vs_t_iter = standard_iterator(numpy_x_0)

    assert np.allclose(x_vs_t_prop, x_vs_t_iter)


def test_stochastic_iterator_vs_propagate(stochastic_iterator, numpy_x_0):

    # Use the iteration wrapper to output a dense timeseries
    seed = stochastic_iterator.rng.uniform_full_int((), dtype=tf.int32)
    x_vs_t_prop = analysis.propagate_x(numpy_x_0, stochastic_iterator.transition_model, stochastic_iterator.n_t, seed=seed)
    stochastic_iterator.rng.reset_from_seed(stochastic_iterator.seed)

    iter_out = stochastic_iterator(numpy_x_0)

    if 'mixed_stochastic_output_iteration_layer' in stochastic_iterator.name:
        # Test the t=0 and t=T batches and extract the stochastic batch
        assert np.allclose(x_vs_t_prop[-1], iter_out[2])
        assert np.allclose(x_vs_t_prop[0], iter_out[0])
        stoch_batch = iter_out[1]
    else:
        stoch_batch = iter_out

    assert x_vs_t_prop.shape[1] == stoch_batch.shape[0]

    # Use broadcasting to compare the stochastic batch to each timepoint
    delta = np.abs(x_vs_t_prop - stoch_batch)
    # There should be at least one timepoint that matches so we sum along time dimension and confirm
    delta_close = np.sum(delta < 1e-11, axis=0)
    assert len(np.unique(delta_close)) >= 1

    # And make sure it's different each call
    stochastic_iterator.rng.reset_from_seed(stochastic_iterator.seed)
    iter_out_2 = stochastic_iterator(numpy_x_0)

    if 'mixed_stochastic_output_iteration_layer' in stochastic_iterator.name:
        # Test the t=0 and t=T batches should be the same
        assert np.allclose(x_vs_t_prop[-1], iter_out_2[2])
        assert np.allclose(x_vs_t_prop[0], iter_out_2[0])
        stoch_batch_2 = iter_out_2[1]
    else:
        stoch_batch_2 = iter_out_2

    # .. but the stochastic batch should be different
    assert not np.allclose(stoch_batch_2, stoch_batch)
