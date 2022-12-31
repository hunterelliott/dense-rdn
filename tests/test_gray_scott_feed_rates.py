import numpy as np
import pytest
import tensorflow as tf

from models.transition import RandomFeedGrayScottTransitionLayer, SinusoidalFeedGrayScottTransitionLayer, \
    GrayScottTransitionLayer, GaussianPulseFeedGrayScottTransitionLayer


def test_random_gray_scott_feed_rate(numpy_x_0):
    """
    Confirms that the feed rate behaves as expected.
    """

    amplitude = .0123  # Test changing some values from defaults
    base_feed_rate = .05
    irreproducible_layer = RandomFeedGrayScottTransitionLayer(amplitude=amplitude, init_feed_rate=base_feed_rate)
    reproducible_layer = RandomFeedGrayScottTransitionLayer(amplitude=amplitude, init_feed_rate=base_feed_rate, seed=314)
    reproduced_reproducible_layer = RandomFeedGrayScottTransitionLayer.from_config(reproducible_layer.get_config())

    # Call the layers to make sure they're built
    irreproducible_layer(numpy_x_0)
    reproducible_layer(numpy_x_0)
    reproduced_reproducible_layer(numpy_x_0)

    # Confirm that the feed rate is random-ish and irreproducible or reproducible as expected, even after serialization
    for i_t in [0, 10, 137]:
        # Layer should either be reproducible or not as specified
        assert not np.allclose(irreproducible_layer.get_feed_rate(i_t), irreproducible_layer.get_feed_rate(i_t))
        assert np.allclose(reproducible_layer.get_feed_rate(i_t), reproducible_layer.get_feed_rate(i_t))

        # Different times should be different
        assert not np.allclose(irreproducible_layer.get_feed_rate(i_t), irreproducible_layer.get_feed_rate(1))
        assert not np.allclose(reproducible_layer.get_feed_rate(i_t), reproducible_layer.get_feed_rate(1))

        # Layer should be reproducible after serialization
        assert np.allclose(reproducible_layer.get_feed_rate(i_t), reproduced_reproducible_layer.get_feed_rate(i_t))

        # And should be in the expected range
        assert -amplitude < reproducible_layer.get_feed_rate(i_t) - base_feed_rate < amplitude
        assert -amplitude < irreproducible_layer.get_feed_rate(i_t) - base_feed_rate < amplitude


def test_random_gray_scott_batch_feed_rate(numpy_x_0):
    """
    Confirms that the feed rate varies per-sample in the batch as expected
    """

    layer = RandomFeedGrayScottTransitionLayer()
    # Call the layer to make sure it's built
    layer(numpy_x_0)

    batch_size = 32
    i_t = np.zeros(shape=(batch_size,))
    feed_rate = layer.get_feed_rate(i_t)

    # Confirm we get the right shape and non-identical feed rates across the batch.
    assert feed_rate.shape[0] == i_t.shape[0]
    assert len(np.unique(feed_rate)) == batch_size

    # And identical starting conditions should give different final values
    x_0_identical = np.ones(shape=(batch_size, 32, 32, 2)) * .1
    x_t = layer(x_0_identical, i_t=123)

    assert not np.allclose(x_t[0], x_t[1])


def test_random_gray_scott_batch_neg_feed_rate(numpy_x_0):
    """
    Tests that negative feed rates will not be returned.
    """
    layer = RandomFeedGrayScottTransitionLayer(amplitude=100.0, seed=111)
    # Call the layer to make sure it's built
    layer(numpy_x_0)

    assert layer.get_feed_rate(0) >= 0


def test_sinusoidal_gray_scott_feed_rate(numpy_x_0):
    """
    Confirms that the feed rate behaves as expected.
    """

    period = 20
    amplitude = .05
    base_feed_rate = .06
    layer = SinusoidalFeedGrayScottTransitionLayer(period=period, amplitude=amplitude, init_feed_rate=base_feed_rate)
    # Call the layer to make sure it's built
    layer(numpy_x_0)

    assert np.allclose(layer.get_feed_rate(0), base_feed_rate)
    assert np.allclose(layer.get_feed_rate(period), base_feed_rate)
    assert np.allclose(layer.get_feed_rate(period / 4), base_feed_rate + amplitude)
    assert np.allclose(layer.get_feed_rate(period * 3 / 4), base_feed_rate - amplitude)


def test_sinusoidal_gray_scott_neg_feed_rate(numpy_x_0):
    """
    Tests that negative feed rates will not be returned irrespective of parameter combinations.
    """
    period = 20
    base_feed_rate = .06
    amplitude = base_feed_rate * 2

    layer = SinusoidalFeedGrayScottTransitionLayer(period=period, amplitude=amplitude, init_feed_rate=base_feed_rate)
    # Call the layer to make sure it's built
    layer(numpy_x_0)

    assert layer.get_feed_rate(3/4 * layer.period / layer.d_t) >= 0


def test_gray_scott_integrate_feed_rate_args(numpy_x_0):
    """
    Test that we can call the integrate method with scalar, vector and matrix feed rates
    """

    layer = GrayScottTransitionLayer()
    x_1_ref_1 = layer(numpy_x_0)
    feed_rate_1 = layer.get_feed_rate(0).numpy()
    feed_rate_2 = layer.get_feed_rate(0) * 1.234
    layer.feed_rate = feed_rate_2
    x_1_ref_2 = layer(numpy_x_0)
    x_0_dc = layer.de_center_x(numpy_x_0)

    # Call with scalar feed rate
    x_1 = layer.integrate(x_0_dc, layer.get_diff_coef(0), feed_rate_1, layer.get_decay_rate(0), layer.d_t)[0]
    assert np.allclose(layer.center_x(x_1), x_1_ref_1)

    # Call with vector feed rate with different values per sample
    batch_size = numpy_x_0.shape[0]
    feed_rate_vec = tf.concat(
        [tf.repeat(feed_rate_1, int(batch_size/2)), tf.repeat(feed_rate_2, int(batch_size/2))], axis=0)
    feed_rate_vec = tf.reshape(feed_rate_vec, (-1, 1, 1))
    x_1 = layer.integrate(x_0_dc, layer.get_diff_coef(0), feed_rate_vec, layer.get_decay_rate(0), layer.d_t)[0]
    x_1 = layer.center_x(x_1)
    # And confirm we get the expected behavior with different sample feed rates
    assert np.allclose(x_1[0], x_1_ref_1[0])
    assert np.allclose(x_1[-1], x_1_ref_2[-1])
    assert not np.allclose(x_1[0], x_1_ref_2[0])

    # Call with matrix feed rate and spatially variable rate
    half_w = int(numpy_x_0.shape[1] / 2)
    feed_rate_mat = np.ones(numpy_x_0.shape[0:3])
    feed_rate_mat[:, 0:half_w, :] = feed_rate_1
    feed_rate_mat[:, half_w:, :] = feed_rate_2
    x_1 = layer.integrate(x_0_dc, layer.get_diff_coef(0), feed_rate_mat, layer.get_decay_rate(0), layer.d_t)[0]
    x_1 = layer.center_x(x_1)
    # And confirm behavior varies appropriately in space
    assert np.allclose(x_1[:, 0:half_w, :], x_1_ref_1[:, 0:half_w, :])
    assert np.allclose(x_1[:, half_w:, :], x_1_ref_2[:, half_w:, :])
    assert not np.allclose(x_1[:, half_w:, :], x_1_ref_1[:, half_w:, :])


def test_gray_scott_spatial_feed_rate_signature(numpy_x_0):
    # TODO - make fixture, test all/both layers!!!!!
    layer = GaussianPulseFeedGrayScottTransitionLayer()

    # Call it once to initialize
    layer(numpy_x_0)

    # Call with scalar time, make sure we get back an appropriately shaped feed tensor
    f = layer.get_feed_rate(0)
    assert f.shape == numpy_x_0.shape[0:3]
    assert f.dtype == tf.float32

    # Call with vector time, make sure we get back an appropriately shaped feed tensor
    f = layer.get_feed_rate(np.ones(numpy_x_0.shape[0]).astype(np.int32))
    assert f.shape == numpy_x_0.shape[0:3]
    assert f.dtype == tf.float32


@pytest.fixture(params=[{},
                        {'pulse_period': 1,  'max_n_t': 16},
                        {'pulse_period': 64, 'max_n_t': 64},
                        {'pulse_spacing': 3},
                        {'pulse_spacing': 4}])
def gauss_pulse_layer(request):
    return GaussianPulseFeedGrayScottTransitionLayer(**request.param)


def test_gauss_pulse_positions_sanity(numpy_x_0, gauss_pulse_layer, seed):
    """
    Test that the number and spatiotemporal positioning of the pulses is as expected
    """

    batch_size = numpy_x_0.shape[0]
    gauss_pulse_layer(numpy_x_0)  # Call to initialize

    pulse_t, pulse_x, pulse_y = gauss_pulse_layer.get_pulse_positions(batch_size, seed)

    # Make sure we got the right number of pulses
    n_expected = int(gauss_pulse_layer.max_n_t / gauss_pulse_layer.pulse_period_it)
    assert pulse_t.shape == (batch_size, n_expected)
    assert pulse_x.shape == (batch_size, n_expected)
    assert pulse_y.shape == (batch_size, n_expected)

    # Check that the times make sense
    assert np.all(pulse_t >= 0) and np.all(pulse_t < gauss_pulse_layer.max_n_t)

    # And the positions are sane
    assert np.all(pulse_x >= 0) and np.all(pulse_x < numpy_x_0.shape[2])
    assert np.all(pulse_y >= 0) and np.all(pulse_y < numpy_x_0.shape[1])

    # We expect integer positions
    assert np.all(pulse_x % 1 == 0) and np.all(pulse_y % 1 == 0) and np.all(pulse_t % 1 == 0)

    # Positions should follow the specified spacing
    assert np.all(pulse_x % gauss_pulse_layer.pulse_spacing == 0)
    assert np.all(pulse_y % gauss_pulse_layer.pulse_spacing == 0)

    # ... because one time I messed up the seed inputs, confirm they're not repeated
    assert not np.all(pulse_x == pulse_y)
    assert not np.all(pulse_t == pulse_y)

    # Verify that different seeds give different results
    pulse_t1, pulse_x1, pulse_y1 = gauss_pulse_layer.get_pulse_positions(batch_size, seed + 1)
    assert not np.all(pulse_t1 == pulse_t)
    assert not np.all(pulse_x1 == pulse_x)
    assert not np.all(pulse_y1 == pulse_y)

    # ... and that the same seed gives the same results
    pulse_t1, pulse_x1, pulse_y1 = gauss_pulse_layer.get_pulse_positions(batch_size, seed)
    assert np.all(pulse_t1 == pulse_t)
    assert np.all(pulse_x1 == pulse_x)
    assert np.all(pulse_y1 == pulse_y)


def mock_get_pulse_positions(batch_size, seed):

    pulse_t = np.repeat(np.reshape((seed, seed + 16), (1, 2)), batch_size, axis=0).astype(np.int32)
    pulse_x = np.repeat(np.reshape((8, 16), (1, 2)), batch_size, axis=0).astype(np.int32)
    pulse_y = np.repeat(np.reshape((12, 6), (1, 2)), batch_size, axis=0).astype(np.int32)

    return pulse_t, pulse_x, pulse_y


def test_gauss_pulse_feed_structure(numpy_x_0):
    """
    Do some simple checks that the actual generated feed matches the pulse positions
    """
    amp = 1.23
    sig = 2
    layer = GaussianPulseFeedGrayScottTransitionLayer(pulse_period=16, max_n_t=32,
                                                      pulse_sigma_xy=sig, pulse_sigma_t=sig,
                                                      pulse_amplitude=amp)
    # Get predictable contrived behavior for testing
    layer.get_pulse_positions = mock_get_pulse_positions
    pulse_t, pulse_x, pulse_y = mock_get_pulse_positions(numpy_x_0.shape[0], 10)

    # Call to initialize
    layer(numpy_x_0)

    # With this mock pulse position generator this should center us (in time) on the first pulse
    f = layer.get_feed_rate(10, 10)
    # Confirm that the pulse amplitude is as expected
    np.allclose(tf.reduce_max(f), layer.feed_rate + layer.pulse_amplitude)
    # And the minimum should match the baseline
    assert tf.reduce_min(f) == layer.feed_rate
    # This should be true at the second pulse as well
    f = layer.get_feed_rate(26, 10)
    np.allclose(tf.reduce_max(f), layer.feed_rate + layer.pulse_amplitude)

    # Now get the feed rate far away in time from the pulses
    f = layer.get_feed_rate(1000, 10)
    # and confirm we're back to baseline feed
    assert tf.reduce_max(f) == layer.feed_rate

    # Prepare a gaussian profile of the expected shape & amplitude/offset
    from ops import gaussian_kernel_1d
    k = gaussian_kernel_1d(sig)
    k = k / np.amax(k) * amp + layer.feed_rate
    w = int((k.shape[0] - 1) / 2)

    # Now confirm the positions and shape match
    for i_pulse in range(2):
        f = layer.get_feed_rate(10 + 16*i_pulse, 10)
        f_max_ij = np.unravel_index(np.argmax(f[0]), f[0].shape)
        assert f_max_ij == (pulse_y[0, i_pulse], pulse_x[0, i_pulse])

        # Extract spatial profiles around the peaks and compare them to the reference
        x_profile = f[0, f_max_ij[0], f_max_ij[1] - w:f_max_ij[1] + w + 1]
        y_profile = f[0, f_max_ij[0]-w:f_max_ij[0]+w+1, f_max_ij[1]]

        assert np.allclose(k, x_profile) and np.allclose(k, y_profile)

    # Finally we confirm that the temporal profile is gaussian as expected
    t_profile = np.zeros(k.shape[0])
    for i_t in range(k.shape[0]):
        f = layer.get_feed_rate(10 - w + i_t, 10)
        t_profile[i_t] = f[0, pulse_y[0, 0], pulse_x[0, 0]]

    assert np.allclose(t_profile, k)
