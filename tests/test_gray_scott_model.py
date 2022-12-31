import pytest
import numpy as np

from models.transition import GrayScottTransitionLayer, VariableFeedGrayScottTransitionLayer, \
    SinusoidalFeedGrayScottTransitionLayer, RandomFeedGrayScottTransitionLayer, \
    GaussianGridFeedGrayScottTransitionLayer, GaussianPulseFeedGrayScottTransitionLayer
from tensorflow.keras.models import Model


@pytest.fixture(params=[GrayScottTransitionLayer, VariableFeedGrayScottTransitionLayer,
                        SinusoidalFeedGrayScottTransitionLayer, RandomFeedGrayScottTransitionLayer,
                        GaussianGridFeedGrayScottTransitionLayer, GaussianPulseFeedGrayScottTransitionLayer])
def gray_scott_layer(request):
    return request.param


non_deterministic_feed_layers = (RandomFeedGrayScottTransitionLayer, GaussianGridFeedGrayScottTransitionLayer,
                                 GaussianPulseFeedGrayScottTransitionLayer)


def test_gray_scott_neg_conc(gray_scott_layer):
    """
    Integration errors can produce negative concentrations so we test for the clipping of those
    """

    layer = gray_scott_layer()
    x_0 = np.zeros((4, 32, 32, 2))
    x_0[:, 8:24, 8:24, :] = -1.1

    x_t = layer(x_0)

    assert np.amin(x_t) >= -1  # Layer works in centered (-1,1) range


def test_non_fittable_gray_scott(gray_scott_layer, numpy_x_0, input_x_0):
    """
    Tests that when parameters are not fittable they don't change on .fit call.
    """
    init_decay_rate = .05  # Change one parameter from default
    layer = gray_scott_layer(init_decay_rate=init_decay_rate, fit_params=())

    x_t = layer(input_x_0)
    model = Model(inputs=[input_x_0], outputs=[x_t])
    model.compile(optimizer='sgd', loss='mse')

    assert len(model.trainable_weights) == 0

    init_diff_coef = layer.get_diff_coef(0).numpy()
    if not isinstance(layer, non_deterministic_feed_layers):
        init_feed_rate = layer.get_feed_rate(0).numpy()
    else:
        init_feed_rate = layer.feed_rate.numpy()

    model.fit(numpy_x_0, numpy_x_0)

    assert model.get_layer(layer.name).get_decay_rate(0).numpy() == init_decay_rate
    if not isinstance(layer, non_deterministic_feed_layers):
        assert model.get_layer(layer.name).get_feed_rate(0).numpy() == init_feed_rate
    else:
        assert model.get_layer(layer.name).feed_rate.numpy() == init_feed_rate
    assert model.get_layer(layer.name).get_diff_coef(0)[0].numpy() == init_diff_coef[0]


def test_fittable_gray_scott(gray_scott_layer, numpy_x_0, input_x_0):
    """
    Tests that when parameters are fittable they change on .fit call.
    """

    init_decay_rate = .05
    layer = gray_scott_layer(init_decay_rate=init_decay_rate, fit_params=('init_decay_rate', 'init_feed_rate'))

    x_t = layer(input_x_0)
    model = Model(inputs=[input_x_0], outputs=[x_t])
    model.compile(optimizer='sgd', loss='mse')

    assert len(model.trainable_weights) == 2

    init_feed_rate = layer.feed_rate.numpy()
    model.fit(numpy_x_0, numpy_x_0)

    assert not np.allclose(model.get_layer(layer.name).decay_rate.numpy(), init_decay_rate)
    assert not np.allclose(model.get_layer(layer.name).feed_rate.numpy(), init_feed_rate)

    layer = gray_scott_layer(init_decay_rate=init_decay_rate, fit_params='all')
    x_t = layer(input_x_0)
    model = Model(inputs=[input_x_0], outputs=[x_t])
    model.compile(optimizer='sgd', loss='mse')

    assert len(model.trainable_weights) == 3

    init_diff_coef = layer.get_diff_coef(0).numpy()
    init_feed_rate = layer.feed_rate.numpy()

    model.fit(numpy_x_0, numpy_x_0)

    assert not np.allclose(layer.get_decay_rate(0).numpy(), init_decay_rate)
    assert not np.allclose(layer.feed_rate.numpy(), init_feed_rate)
    assert not np.allclose(layer.get_diff_coef(0).numpy(), init_diff_coef)


def test_fittable_gray_scott_fit_arg():
    """
    Make sure that the layer handles inputs as expected
    """

    # Confirm that exception is raised so typos in fit_params don't result in silent failures to fit.
    with pytest.raises(ValueError) as e_info:
        layer = GrayScottTransitionLayer(fit_params=('bla_bla', 'bla'))
    with pytest.raises(ValueError) as e_info:
        layer = GrayScottTransitionLayer(fit_params='bla_bla')

    layer = GrayScottTransitionLayer(fit_params=())
    layer = GrayScottTransitionLayer(fit_params='all')
    layer = GrayScottTransitionLayer(fit_params='init_decay_rate')
    layer = GrayScottTransitionLayer(fit_params=('init_decay_rate', 'init_feed_rate'))


