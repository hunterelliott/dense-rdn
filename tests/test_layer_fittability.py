"""
Tests that the various flags that control parameter fittability / trainability work as expected
"""

import models
import layers
import pytest

from tensorflow.keras.layers import serialize, deserialize


@pytest.fixture(params=[["FlowDriveLayer",
                        {'init_feed_conc': (1.0, 0.2), 'init_flow_rate': 0.2},
                        {'fit_feed_conc': False, 'fit_flow_rate': False}],
                        ["ConstantSynthesisDriveLayer",
                        {'init_synth_rate': (.2, .1)},
                        {'fit_synth_rate': False}],
                        ["ReactionNetworkTransitionLayer",
                         {'reactants': [[1, 0],[0, 1]], 'products': [[0, 1], [1, 0]],
                          'init_rate_const': (.2, .3), 'init_diff_coeffs': (2.3, 3.4)},
                         {'fit_rate_const': False, 'fit_diff_coeffs': False}]])
def optionally_fittable_class(request):
    if hasattr(models, request.param[0]):
        layer_class = getattr(models, request.param[0])
    else:
        layer_class = getattr(models.transition, request.param[0])

    return layer_class, request.param[1], request.param[2]


def test_layer_fit_toggle(numpy_x_0, optionally_fittable_class):

    layer_class, args, fit_params = optionally_fittable_class

    # Test each param separately
    for param in fit_params:
        # Set just this argument to true, confirm we get one fittable param
        fit_params[param] = True
        layer_instance = layer_class(**args, **fit_params)
        # Call to make sure build method has been called
        layer_instance(numpy_x_0)
        assert len(layer_instance.trainable_weights) == 1
        assert param[4:] in layer_instance.trainable_weights[0].name

        # And make sure this is true after serialization
        layer_recon = deserialize(serialize(layer_instance))
        layer_recon(numpy_x_0)
        assert len(layer_instance.trainable_weights) == 1

        fit_params[param] = False

    # Test all at once
    for param in fit_params:
        fit_params[param] = True
    layer_instance = layer_class(**args, **fit_params)
    # Call to make sure build method has been called
    layer_instance(numpy_x_0)
    assert len(layer_instance.trainable_weights) == len(fit_params)








