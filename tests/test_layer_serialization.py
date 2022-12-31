import copy
import numpy as np
import pytest

import tensorflow as tf
from tensorflow.keras.layers import serialize, deserialize

import layers
import models


@pytest.fixture(params=[['RescalingLayer', {'init_scale': 1.11}],
                        ['ChannelOffsetLayer', {'i_chan': 1, 'init_offset': 0.234}],
                        ['ClippingLayer', {'out_range': (-.01, 1.23)}],
                        ['ConstantSynthesisDriveLayer', {'init_synth_rate': (1.1, 2.3)}],
                        ['FlowDriveLayer', {'init_feed_conc': (.3, .2),
                                            'init_flow_rate': .123, 'fit_feed_conc': True}],
                        ['NoisyFlowDriveLayer', {'init_feed_conc': (.3, .2),
                                                 'init_flow_rate': .123,
                                                 'noise_amplitude': .1122}]])
def misc_layer(request):
    if hasattr(layers, request.param[0]):
        return getattr(layers, request.param[0])(**request.param[1])
    else:
        return getattr(models, request.param[0])(**request.param[1])


def test_serialize_deserialize_propagation_layers(iterator, numpy_x_0):
    """
    Tests that transition models embedded in iterators serialzie and deserialize properly.
    """

    layer_serial = serialize(iterator)
    layer_reconstruct = deserialize(copy.deepcopy(layer_serial))

    layer_re_serial = serialize(layer_reconstruct)

    if layer_serial['config']['transition_model']['class_name'] == 'Iteration>VariableFeedGrayScottTransitionLayer':
        # We can't rely on Lambda function serialized forms matching so exclude that
        layer_serial['config']['transition_model']['config']['feed_rate_func']['config']['function'] = 'ignore'
        layer_re_serial['config']['transition_model']['config']['feed_rate_func']['config']['function'] = 'ignore'

    np.testing.assert_equal(layer_serial, layer_re_serial)

    # Pass some data through and confirm it matches, setting the seed for stochastic layers
    tf.random.get_global_generator().reset_from_seed(44)
    tf.random.set_seed(44)
    x_t = iterator(numpy_x_0)
    tf.random.get_global_generator().reset_from_seed(44)
    tf.random.set_seed(44)
    x_t_reconstruct = layer_reconstruct(numpy_x_0)

    assert np.allclose(x_t, x_t_reconstruct)


def test_serialize_deserialize_misc_layers(misc_layer, numpy_x_0):
    """
    Tests that misc other layers serialize/deserialize properly
    """


    layer_serial = serialize(misc_layer)
    layer_reconstruct = deserialize(copy.deepcopy(layer_serial))

    layer_re_serial = serialize(layer_reconstruct)

    np.testing.assert_equal(layer_serial, layer_re_serial)

    # Pass some data through and confirm it matches
    tf.random.set_seed(111)
    x_t = misc_layer(numpy_x_0)
    tf.random.set_seed(111)
    x_t_reconstruct = layer_reconstruct(numpy_x_0)

    assert np.allclose(x_t, x_t_reconstruct)
