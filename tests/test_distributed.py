"""
Some tests of distributed (multi-GPU) execution, to the extent that's reasonable...
"""
import tensorflow as tf
from tensorflow.keras.layers import Lambda


def confirm_multiple_devices():
    logical_devices = tf.config.list_logical_devices('CPU')
    assert len(logical_devices) > 1


def test_iteration_layer_seed_gen(standard_iterator_class):

    dist_strat = tf.distribute.MirroredStrategy()
    assert dist_strat.num_replicas_in_sync > 1

    # Initialize the global generator outside the scope.
    tf.random.get_global_generator()

    with dist_strat.scope():

        transition_model = Lambda(lambda x: x * 1.1)
        layer = standard_iterator_class(transition_model, 4)

        rand_ints = dist_strat.run(lambda: layer.rng.uniform_full_int((), dtype=tf.int32))

        # Make the layers on each device are giving different seeds
        assert not rand_ints.values[0].numpy() == rand_ints.values[1].numpy()

        # ... and that each call is giving different numbers.
        rand_ints_2 = dist_strat.run(lambda: layer.rng.uniform_full_int((), dtype=tf.int32))

        assert not rand_ints_2.values[0].numpy() == rand_ints.values[0].numpy()