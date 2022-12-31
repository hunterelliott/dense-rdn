import pytest
import numpy as np

import tensorflow as tf
# Create some virtual devices for testing on CPU
physical_devices = tf.config.list_physical_devices('CPU')
tf.config.set_logical_device_configuration(physical_devices[0],
                                           [tf.config.LogicalDeviceConfiguration(),
                                            tf.config.LogicalDeviceConfiguration()])


from tensorflow.keras.layers import Input

import models
import layers


@pytest.fixture(params=[["GrayScottTransitionLayer", {'init_decay_rate': .01}],
                        ["VariableFeedGrayScottTransitionLayer", {}],
                        ["SinusoidalFeedGrayScottTransitionLayer", {'init_diff_coef': (.25, .15)}],
                        ["RandomFeedGrayScottTransitionLayer", {'seed': 137}],
                        ["GrayScottTransitionLayer", {'init_feed_rate':  .045,
                                                      'fit_params': ('init_decay_rate', 'init_diff_coef'),
                                                      'reaction_entropy_loss_weight': -1.0}],
                        ["GaussianGridFeedGrayScottTransitionLayer", {'pulse_frequency': 2,
                                                                      'pulse_sigma_t': 1,
                                                                      'max_n_t': 16}],
                        ["GaussianPulseFeedGrayScottTransitionLayer", {'pulse_period': 2,
                                                                       'pulse_sigma_t': 1,
                                                                       'max_n_t': 16}],
                        ["BrusselatorTransitionLayer", {'a': 1.1, 'b': 2.3, 'd_x': 1.5, 'd_y': 2.4, 'd_t': .01}],
                        ["ReactionNetworkTransitionLayer", {'reactants': np.array([[1, 0, 0],[2, 1, 3]]),
                                                            'products': np.array([[0, 0, 1],[3, 0, 2]]),
                                                            'init_rate_const': np.array([.5, .05, .8]),
                                                            'init_diff_coeffs': np.array([.2, .15]),
                                                            'fit_diff_coeffs': True,
                                                            'drive': models.drive.FlowDriveLayer(
                                                                init_feed_conc=np.array([.7, 0.01]), init_flow_rate=.1),
                                                            'd_t':.2}]])
def transition_model(request):
    return getattr(models.transition, request.param[0])(**request.param[1])


@pytest.fixture(params=[[models.transition.GrayScottTransitionLayer, {}],
                        [models.transition.ReactionNetworkTransitionLayer, {'reactants': np.eye(2),
                                                                            'products': np.flip(np.eye(2), 0)}]])
def therm_transition_layer_class(request):
    return request.param


@pytest.fixture
def numpy_x_0():
    # TODO - make dependent on transition layer
    x_0 = np.random.uniform(-.5, .5, size=(4, 32, 32, 2)).astype(np.float32)
    # Make sure we test handling zero-concentration inputs
    x_0[:, 0:8, 0:8, :] = -1
    return x_0


@pytest.fixture
def tiny_numpy_x_0():
    # TODO - make dependent on transition layer
    tiny_numpy_x_0 = np.random.uniform(-.5, .5, size=(2, 16, 16, 2)).astype(np.float32)
    # Make sure we test handling zero-concentration inputs
    tiny_numpy_x_0[:, 0:5, 0:5, :] = -1
    return tiny_numpy_x_0


@pytest.fixture(params=[2, 5]) #I don't test 1 because I'm never using it and it'll probably break stuff...
def n_chans(request):
    return request.param


@pytest.fixture
def n_chan_numpy_x_0(n_chans):
    # For testing functions which should accept n-channel/n-species inputs
    x_0 = np.random.uniform(-.5, .5, size=(4, 32, 32, n_chans)).astype(np.float32)
    # Make sure we test handling zero-concentration inputs
    x_0[:, 0:8, 0:8, :] = -1
    return x_0


@pytest.fixture
def numpy_x_t():
    # TODO - make dependent on transition layer
    return np.random.uniform(-.5, .5, size=(4, 32, 32, 2)).astype(np.float32)


@pytest.fixture
def input_x_0():
    # TODO - make dependent on transition layer
    return Input(shape=(32, 32, 2), batch_size=4)


@pytest.fixture(params=["StochasticOutputIterationLayer", "MixedStochasticOutputIterationLayer"])
def stochastic_iterator_class(request):
    return getattr(layers, request.param)


@pytest.fixture
def stochastic_iterator(stochastic_iterator_class, transition_model):
    n_t = 8
    return stochastic_iterator_class(transition_model, n_t, x_output_period=int(n_t/8), seed=313)


@pytest.fixture(params=["IterationLayer", "SymbolicIterationLayer"])
def standard_iterator_class(request):
    return getattr(layers, request.param)


@pytest.fixture
def standard_iterator(standard_iterator_class, transition_model):
    n_t = 8
    return standard_iterator_class(transition_model, n_t, x_output_period=int(n_t/4), seed=314)


@pytest.fixture(params=["IterationLayer", "SymbolicIterationLayer",
                        "StochasticOutputIterationLayer", "MixedStochasticOutputIterationLayer"])
def iterator_class(request):
    return getattr(layers, request.param)


@pytest.fixture(params=["IterationLayer", "StochasticOutputIterationLayer", "MixedStochasticOutputIterationLayer"])
def non_symbolic_iterator_class(request):
    return getattr(layers, request.param)


@pytest.fixture
def iterator(iterator_class, transition_model):
    n_t = 8
    return iterator_class(transition_model, n_t, x_output_period=int(n_t/4))


@pytest.fixture(params=[1, 2])  # Just so we use at least a couple different randoms
def seed():
    rng = np.random.default_rng()
    return rng.integers(1, int(1e9), dtype=np.int32)
