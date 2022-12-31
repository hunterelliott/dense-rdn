"""
Tests of various simple arithmetic/utility/etc. layers
"""
import pytest
import numpy as np
import layers


@pytest.fixture(params=[[1], [0, 1], [0]])
def i_chan(request):
    return request.param


def test_chan_offset_layer(numpy_x_0, i_chan):

    offset = np.random.uniform(-2, 2, size=len(i_chan))
    layer = layers.ChannelOffsetLayer(i_chan=i_chan, init_offset=offset)

    x_off = layer(numpy_x_0).numpy()

    # Confirm the weight is trainable
    assert len(layer.trainable_weights) == 1

    # And that it behaves as expected - only the specified channel is offset
    offset = np.reshape(offset, [1, 1, 1, len(i_chan)])
    assert np.allclose(numpy_x_0[:, :, :, i_chan] + offset, x_off[:, :, :, i_chan])

    # And un-offset channels aren't offset
    i_chan_comp = np.setdiff1d(np.arange(0, numpy_x_0.shape[-1]), i_chan)
    assert np.allclose(numpy_x_0[:, :, :, i_chan_comp], x_off[:, :, :, i_chan_comp])


def test_scaling_layer(numpy_x_0):

    scale = 1.21
    layer = layers.RescalingLayer(init_scale=scale)

    x_off = layer(numpy_x_0)

    # Confirm the weight is trainable
    assert len(layer.trainable_weights) == 1

    # And that it behaves as expected
    assert np.allclose(numpy_x_0 * scale, x_off)


def test_clip_layer(numpy_x_0):

    out_range = (-.001, np.amax(numpy_x_0)+1)
    layer = layers.ClippingLayer(out_range=out_range)

    x_clip = layer(numpy_x_0)

    assert len(layer.trainable_weights) == 0

    # Confirm expected behavior
    assert np.allclose(np.amin(x_clip), out_range[0])
    assert np.amax(x_clip) == np.amax(numpy_x_0)

    out_range = (-.001, .002)
    layer = layers.ClippingLayer(out_range=out_range)
    x_clip = layer(numpy_x_0)
    assert np.allclose(np.amax(x_clip), out_range[1])
