import pytest
import numpy as np
import tensorflow as tf

from therm import diffusion_entropy_rate, irreversible_reaction_entropy_rate, gray_scott_reaction_entropy_rate
from ops import laplacian_kernel, pad_toroidal
import layers


def test_diff_entropy_rate_sanity():
    """
    Runs some simple sanity checks on the diffusion entropy rate calc function
    """

    batch_size = 8
    c_unif = np.ones((batch_size, 32, 32, 3))

    # Uniform concentrations should have no diffusion entropy production
    assert np.allclose(diffusion_entropy_rate(c_unif, (1.3, 3.4, 5.6)), np.zeros(batch_size))

    c_delta = np.zeros((batch_size, 16, 16, 2)).astype(np.float32) + .01
    c_delta[:, 7, 7, :] = 0.5

    diff_coef = (.1, .2)
    delta_s = diffusion_entropy_rate(c_delta, diff_coef)
    delta_s_1 = diffusion_entropy_rate(tf.expand_dims(c_delta[:, :, :, 0], -1), diff_coef[0])
    delta_s_2 = diffusion_entropy_rate(tf.expand_dims(c_delta[:, :, :, 1], -1), diff_coef[1])

    # All else equal should go up with diffusion coefficient
    assert np.all(delta_s_1 < delta_s_2)

    # Should sum across species
    assert np.allclose(delta_s, delta_s_1 + delta_s_2)

    # Perform one time step of diffusion
    c_delta_t = c_delta + tf.nn.depthwise_conv2d(pad_toroidal(c_delta, 1), laplacian_kernel(2), [1, 1, 1, 1], 'VALID') \
        * tf.expand_dims(diff_coef, axis=0)

    # And confirm that the rate decreases with time
    delta_s_t = diffusion_entropy_rate(c_delta_t, diff_coef)
    delta_s_1_t = diffusion_entropy_rate(tf.expand_dims(c_delta_t[:, :, :, 0], -1), diff_coef[0])
    delta_s_2_t = diffusion_entropy_rate(tf.expand_dims(c_delta_t[:, :, :, 1], -1), diff_coef[1])

    assert np.all(delta_s_t < delta_s)
    assert np.all(delta_s_1_t < delta_s_1)
    assert np.all(delta_s_2_t < delta_s_2)


def test_layer_no_loss(therm_transition_layer_class, numpy_x_0):
    therm_transition_layer_class, args = therm_transition_layer_class

    # Default behavior is to have no loss terms
    layer = therm_transition_layer_class(**args)
    x_t = layer(numpy_x_0)

    assert len(layer.losses) == 0


def test_add_layer_diff_entropy_loss(therm_transition_layer_class, numpy_x_0):
    therm_transition_layer_class, args = therm_transition_layer_class

    layer = therm_transition_layer_class(**args, diffusion_entropy_loss_weight=1.0)

    x_t = layer(numpy_x_0)

    assert len(layer.losses) == 1
    assert layer.losses[0].shape == ()

    ds_diff = np.mean(diffusion_entropy_rate(layer.de_center_x(numpy_x_0), layer.diff_coeffs.numpy()) * layer.d_t)
    assert np.allclose(layer.losses[0], layer.scale_diffusion_loss(ds_diff))


def test_add_functional_model_diff_entropy_loss(therm_transition_layer_class, numpy_x_0):
    therm_transition_layer_class, args = therm_transition_layer_class

    layer = therm_transition_layer_class(**args, diffusion_entropy_loss_weight=1.0)

    x_in = tf.keras.layers.Input(shape=numpy_x_0.shape[1::], batch_size=numpy_x_0.shape[0])
    x_out = layer(layer(x_in))

    model = tf.keras.Model(inputs=[x_in], outputs=[x_out])

    x_t = model(numpy_x_0)

    # Confirm we are retaining the loss for each timestep
    assert len(model.losses) == 2

    # And that it's cleared on each model-level call
    numpy_x_0 = np.random.uniform(-.01, .01, size=numpy_x_0.shape)
    x_1 = layer(numpy_x_0)  # Loss is for x_t not x_t+1 so this will be needed to match losses.
    x_t = model(numpy_x_0)
    assert len(model.losses) == 2

    # And that they match the xs
    ds_diff_0 = np.mean(diffusion_entropy_rate(layer.de_center_x(numpy_x_0), layer.diff_coeffs.numpy()) * layer.d_t)
    assert np.allclose(layer.losses[0], layer.scale_diffusion_loss(ds_diff_0))
    ds_diff_t = np.mean(diffusion_entropy_rate(layer.de_center_x(x_1), layer.diff_coeffs.numpy()) * layer.d_t)
    assert np.allclose(layer.losses[-1], layer.scale_diffusion_loss(ds_diff_t))


def test_add_iterator_model_diff_entropy_loss(therm_transition_layer_class, iterator_class, numpy_x_0):
    therm_transition_layer_class, args = therm_transition_layer_class

    n_t = 6
    layer = therm_transition_layer_class(**args, diffusion_entropy_loss_weight=1.0)
    iterator_layer = iterator_class(layer, n_t=n_t, x_output_period=3)

    x_in = tf.keras.layers.Input(shape=numpy_x_0.shape[1::], batch_size=numpy_x_0.shape[0])
    x_out = iterator_layer(x_in)

    model = tf.keras.Model(inputs=[x_in], outputs=[x_out])

    x_t = model(numpy_x_0)

    # Confirm we are retaining the loss for each timestep (including t=0)
    assert len(model.losses) == n_t
    # And that each one is a scalar
    assert model.losses[0].shape == ()

    # And that it's cleared on each call
    numpy_x_0 = np.random.uniform(-.01, .01, size=numpy_x_0.shape)
    x_t = model(numpy_x_0)
    assert len(model.losses) == n_t

    # And that they match the xs - compare to simple IterationLayer to handle e.g. stochastic / sub-sampling layer case
    ref_layer = layers.IterationLayer(layer, n_t)
    x_t_ref = ref_layer(numpy_x_0)

    ds_diff_0 = np.mean(diffusion_entropy_rate(layer.de_center_x(numpy_x_0), layer.diff_coeffs.numpy()) * layer.d_t)
    assert np.allclose(layer.losses[0], layer.scale_diffusion_loss(ds_diff_0))
    ds_diff_t = np.mean(diffusion_entropy_rate(layer.de_center_x(x_t_ref[-2]), layer.diff_coeffs.numpy()) * layer.d_t)
    assert np.allclose(layer.losses[-1], layer.scale_diffusion_loss(ds_diff_t))
    assert np.allclose(ref_layer.losses, layer.losses)


def test_rxn_entropy_output_format(numpy_x_0):

    batch_size = numpy_x_0.shape[0]
    ds = irreversible_reaction_entropy_rate([numpy_x_0[:, :, :, 0], numpy_x_0[:, :, :, 0]])

    # We should get a batch size length vector
    assert len(ds.shape) == 1
    assert ds.shape[0] == batch_size


def test_irreversible_rxn_entropy_sanity():
    """
    Runs some simple sanity checks on the reaction entropy production calc
    """

    fwd_rate = 1.23
    v_mat = np.ones((3, 16, 16)) * fwd_rate

    # At equilibrium (v_fwd=v_rev) rate should be zero
    assert np.all(irreversible_reaction_entropy_rate([v_mat], v_rev=fwd_rate) == 0)

    v_mat = np.random.uniform(.1, .9, size=(4, 16, 16))
    v_irr_1 = irreversible_reaction_entropy_rate([v_mat])
    v_irr_2 = irreversible_reaction_entropy_rate([v_mat, v_mat])

    # Twice the reactions, twice the entropy prod rate
    assert np.all(v_irr_1 * 2 == v_irr_2)

    # Twice the reaction rate, more than twice but less than 4x entropy prod rate
    v_irr_2x = irreversible_reaction_entropy_rate([v_mat * 2])
    assert np.all(v_irr_2x > 2*v_irr_1)
    assert np.all(v_irr_2x < 4*v_irr_1)


def test_add_iterator_model_rxn_entropy_loss(therm_transition_layer_class, iterator_class, numpy_x_0):
    therm_transition_layer_class, args = therm_transition_layer_class

    n_t = 6
    layer = therm_transition_layer_class(**args, reaction_entropy_loss_weight=1.0)
    iterator_layer = iterator_class(layer, n_t=n_t, x_output_period=3)

    x_in = tf.keras.layers.Input(shape=numpy_x_0.shape[1::], batch_size=numpy_x_0.shape[0])
    x_out = iterator_layer(x_in)

    model = tf.keras.Model(inputs=[x_in], outputs=[x_out])

    x_t = model(numpy_x_0)

    # Confirm we are retaining the loss for each timestep (including t=0)
    assert len(model.losses) == n_t
    # And that each one is a scalar
    assert model.losses[0].shape == ()

    # And that it's cleared on each call
    numpy_x_0 = np.random.uniform(-.01, .01, size=numpy_x_0.shape)
    x_t = model(numpy_x_0)
    assert len(model.losses) == n_t


def test_add_iterator_model_multi_entropy_loss(therm_transition_layer_class, iterator_class, numpy_x_0):
    therm_transition_layer_class, args = therm_transition_layer_class

    n_t = 6
    layer = therm_transition_layer_class(**args, reaction_entropy_loss_weight=1.0,
                                         diffusion_entropy_loss_weight=1.0)
    iterator_layer = iterator_class(layer, n_t=n_t, x_output_period=3)

    x_in = tf.keras.layers.Input(shape=numpy_x_0.shape[1::], batch_size=numpy_x_0.shape[0])
    x_out = iterator_layer(x_in)

    model = tf.keras.Model(inputs=[x_in], outputs=[x_out])

    x_t = model(numpy_x_0)

    # Confirm we are retaining the loss for each timestep (including t=0)
    assert len(model.losses) == n_t * 2

    # And that it's cleared on each call
    numpy_x_0 = np.random.uniform(-.01, .01, size=numpy_x_0.shape)
    x_t = model(numpy_x_0)
    assert len(model.losses) == n_t * 2
