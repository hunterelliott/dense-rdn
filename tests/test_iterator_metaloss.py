"""
Tests IterationLayer meta loss functionality i.e. from non-None meta_loss_func input
"""
import pytest

import numpy as np
import tensorflow as tf
from therm import dissipation_variation_loss


@pytest.fixture(params=[{'diffusion_entropy_loss_weight': -1.0},
                        {'diffusion_entropy_loss_weight': -1.0, 'reaction_entropy_loss_weight': -1.0}])
def transition_model_with_loss(therm_transition_layer_class, request):
    therm_transition_layer_class, args = therm_transition_layer_class
    return therm_transition_layer_class(**args, **request.param)


def var_meta_loss(losses, n_t):
    # Simple loss for tests
    return tf.math.reduce_variance(tf.stack(losses), name='var_meta_loss')


@pytest.fixture(params=[var_meta_loss, dissipation_variation_loss])
def meta_loss(request):
    return request.param


def test_add_loss(iterator_class, transition_model_with_loss, meta_loss, numpy_x_0):

    n_t = 3
    iterator = iterator_class(transition_model_with_loss, n_t, meta_loss_func=meta_loss)

    x_t = iterator(numpy_x_0)

    assert len(iterator.losses) == len(transition_model_with_loss.losses) + 1

    # And confirm they are appropriately cleared and re-populated on subsequent calls
    x_t = iterator(numpy_x_0)
    assert len(iterator.losses) == len(transition_model_with_loss.losses) + 1

    # And that the value fits the input function
    assert iterator.losses[0] == meta_loss(transition_model_with_loss.losses, n_t)


def test_add_loss_functional_model(non_symbolic_iterator_class, transition_model_with_loss, meta_loss, numpy_x_0):

    n_t = 3
    iterator = non_symbolic_iterator_class(transition_model_with_loss, n_t, meta_loss_func=meta_loss)
    x_0 = tf.keras.layers.Input(shape=numpy_x_0.shape[1:], batch_size=numpy_x_0.shape[0])
    x_t = iterator(x_0)
    model = tf.keras.Model(inputs=x_0, outputs=x_t)

    x_vs_t = model(numpy_x_0)
    # Check that we have the right number of losses and that this remains true on repeat calls
    assert len(model.losses) == len(transition_model_with_loss.losses) + 1
    x_vs_t = model(numpy_x_0)
    assert len(model.losses) == len(transition_model_with_loss.losses) + 1

    # Confirm the value makes sense
    assert model.losses[0] == meta_loss(transition_model_with_loss.losses, n_t)


def test_diss_var_loss(non_symbolic_iterator_class, transition_model_with_loss, numpy_x_0):

    n_groups = 0
    loss_weights = ['reaction_entropy_loss_weight', 'diffusion_entropy_loss_weight']
    loss_weights_present = []
    for loss_wt in loss_weights:
        if getattr(transition_model_with_loss, loss_wt) != 0:
            n_groups = n_groups + 1
            loss_weights_present.append(loss_wt)

    # Build a model so we can use graph mode which preserves op names on our losses
    n_t = 5
    loss_func = lambda loss, n_t : dissipation_variation_loss(loss, n_t, n_loss_groups=n_groups, is_scaled_by_nt=False)
    iterator = non_symbolic_iterator_class(transition_model_with_loss, n_t, meta_loss_func=loss_func)
    x_0 = tf.keras.layers.Input(shape=numpy_x_0.shape[1:], batch_size=numpy_x_0.shape[0])
    x_t = iterator(x_0)
    model = tf.keras.Model(inputs=x_0, outputs=x_t)

    # Get indices of the losses when op names are present so we can check we're including the right terms
    var_mean = 0
    ind_per_loss = []
    for loss_wt in loss_weights_present:
        ind = []
        for i_loss, loss in enumerate(model.losses):
            if loss_wt[0:17] in loss.op.name:
                ind.append(i_loss)
        ind_per_loss.append(ind)

    # Now pass some data through and confirm the calculation
    x_vs_t = model(numpy_x_0)
    var_mean = 0.0
    for i_group in range(n_groups):
        var_mean = var_mean + np.var(np.array(model.losses)[ind_per_loss[i_group]])

    assert np.allclose(var_mean, model.losses[0])
    assert np.allclose(var_mean, loss_func(transition_model_with_loss.losses, n_t))


@pytest.fixture(params=['var','mad'])
def var_meta_loss_modes(request):
    return request.param[0]


def test_diss_var_loss_n_t_scaling(var_meta_loss_modes):
    """
    Test that when per-timepoint losses are scaled to be averages (1/n_t) the metaloss scales appropriately as well
    """

    n_t = 12
    losses_us = np.random.normal(-1, .23, size=n_t)
    meta_loss_us = dissipation_variation_loss(list(losses_us), n_t, is_scaled_by_nt=False, mode=var_meta_loss_modes)

    losses = list(losses_us / n_t)

    meta_loss = dissipation_variation_loss(losses, n_t, n_loss_groups=1, is_scaled_by_nt=True, mode=var_meta_loss_modes)

    assert np.allclose(meta_loss, meta_loss_us)
    losses = list(np.concatenate([losses_us, losses_us]) / n_t / 2)
    meta_loss_2x = dissipation_variation_loss(losses, n_t, n_loss_groups=1, is_scaled_by_nt=True,
                                              mode=var_meta_loss_modes)

    assert np.allclose(meta_loss, meta_loss_2x)
