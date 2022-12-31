"""
Script for training models which map between Z and X without any time propagation, mostly for testing / control
experiments.
"""

import os
import logging

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

import graphics
import models
import utils
from training import train_iteration_model
from sampling import get_train_z

# ---- Config ---- #


x_shape = (128, 128, 3)
z_shape = (1, 1, 256)
z_bits = 24

x_pad_generator = 32
x_pad_encoder = 32

state_generator = 'simple_generator_model'
state_encoder = 'complement_encoder_model'

train_kwargs = {'batch_size': 48,
                'n_epochs': int(1e4),
                'lr': 1e-3,
                'z_bits': z_bits}

out_dir = '/media/hunter/fast storage/Training_Experiments/Iteration_B1/Z_Mapping_Models/v1_XDomainMapping/test_7'
#import random
#out_dir = '/Users/hunterelliott/TEMP/test_mapping/test_' + str(random.randint(1000, 9999))
#out_dir = '/home/hunter/Desktop/TEMP/test_mapping/test_' + str(random.randint(1000, 9999))

utils.posterity_log(out_dir, locals(), __file__)


# ---- Build Model ---- #

logging.info("Building model...")
dist_strat = tf.distribute.MirroredStrategy()
with dist_strat.scope():
    encoder_model = models.encoder_model(x_shape, z_shape, state_encoder, x_pad=x_pad_encoder)
    generator_model = models.generator_model(x_shape, z_shape, state_generator, x_pad=x_pad_generator)

    z = Input(shape=z_shape)
    x = generator_model(z)
    z_prime = encoder_model(x)

    model = Model(inputs=z, outputs=z_prime, name='mapping_model')


# ---- Train ---- #

train_iteration_model(model, out_dir, **train_kwargs)


# ---- Output ---- #

logging.info("Saving output...")
utils.save_model(model, out_dir)

z_shape = model.input.shape[1::]
z_test = get_train_z(z_shape, train_kwargs['batch_size'], entropy=z_bits).astype(np.float32)
x_test = generator_model(z_test, training=True)
fig = graphics.show_x_of_t_batch(np.expand_dims(x_test, axis=0), 0)
fig.savefig(os.path.join(out_dir, 'X.png'))
logging.info("Done!")
