"""
Iterates a model in a spatiotemporal domain larger than used for optimization.
"""
import os
import logging

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding2D, GaussianNoise

import graphics
import models
import utils
from analysis import get_compound_model_components
from sampling import get_train_z

x_scale_up_factor = (1, 8, 1)  # Iteration domain will be larger by this factor in (height, width) than the original parent X.
#n_t = 48 * 4
z_bits = 24
batch_size_gen = 40  # batch size for X_0 generation (can be different to e.g. improve batch norm statistics)
batch_size_extrapolate = batch_size_gen
use_training = True
preproc_gen_t0 = True  # If true, apply the preprocessing layer to each generation's t=0

model_dir = '/Users/hunterelliott/Iteration_B1/Z_Replication_Models/v8_2en_XvsT_ConvDisplacement/test_19'
import random
out_dir = os.path.join(model_dir, 'extrapolation_' + str(random.randint(1000, 9999)))
# TODO - setup resizeable models / store these in them....
#transition_kwargs = {'batch_size': batch_size_extrapolate}

utils.posterity_log(out_dir, locals(), __file__)


# ---- Get / Reconfigure Model ---- #

model = utils.load_model(model_dir)


# Workaround for problem with granddaughter models...
# TODO - TEMP WORKAROUND!!! FIX THIS SHIT IT IS ERROR-PRONE!!
x_shape = (32, 32, 3)
z_shape = (1, 1, 256)

x_buffer = 16

n_t = 48

use_x_similarity_loss = True
use_z_reconstruction_loss = False

transition_kwargs = {'batch_size': batch_size_extrapolate}
generator_kwargs = {}
# generator_kwargs = {}
build_kwargs = {'state_generator': 'simple_generator_model',
                'state_encoder': 'simple_encoder_model',
                'state_transition_model': 'convolutional_displacement_transition_model',
                'batch_size': batch_size_extrapolate,
                'x_buffer': x_buffer,
                'x_yard': x_shape[0] + 2*x_buffer,
                'x_outputs': use_x_similarity_loss,
                'z_outputs': use_z_reconstruction_loss,
                'granddaughter_fraction': 0.5,
                'n_t_x_loss': 20,
                'x_0_preproc_layer': GaussianNoise(0.0125)
                }

#wts = model.get_weights()
vars_og = model.variables
var_names_og = [v.name for v in vars_og]
model = models.iteration.replication_model(x_shape, z_shape, n_t, **build_kwargs,
                                           transition_kwargs=transition_kwargs, generator_kwargs=generator_kwargs)
#model.set_weights(wts)
# Do it this way as a workaround for the bug causing some weights to be duplicated on loading
for var in model.variables:
    var.assign(vars_og[var_names_og.index(var.name)].value())

n_t_og = n_t
n_t = n_t * 4

generator_model, transition_model_og, iteration_layer, encoder_model = get_compound_model_components(model)
i_out = model.output_names.index('x_output')
x_hat_shape = model.outputs[i_out].shape[2:5]
x_shape = generator_model.output.shape[1:]
x_full_shape = [a * b for a, b in zip(x_hat_shape, x_scale_up_factor)]
x_pad_x_out = tuple([(int((a-b)/2), int((a-b)/2)) for a, b in zip(x_full_shape, x_hat_shape)])[0:2]

# Currently, we have to create a new transition model and copy weights over.
transition_model = models.transition_model(x_full_shape, transition_model_og.name, **transition_kwargs)
transition_model.set_weights(transition_model_og.get_weights())

# ---- Get X0 and iterate  ----- #

logging.info("Iterating...")
z_0 = get_train_z(model.input.shape[1:], batch_size_gen, entropy=z_bits)
model_out = model(z_0, training=use_training)[0]
if len(model.outputs) > 1:
    x_hat_0 = model_out[i_out][0:batch_size_extrapolate, :, :, :, 0]
else:
    x_hat_0 = model_out[0:batch_size_extrapolate, :, :, :, 0]
x_full_0 = ZeroPadding2D(x_pad_x_out)(x_hat_0[0:batch_size_extrapolate])

# TODO - move out of RAM! (write as you go/virtual array/something)
x_full_vs_t = np.zeros((n_t+1,) + x_full_0.shape)
x_full_t = x_full_0
for i_t in range(n_t+1):

    if build_kwargs['x_0_preproc_layer'] is not None and preproc_gen_t0 and i_t % n_t_og == 0:
        logging.info("Applying pre-processing layer to t={}".format(i_t))
        x_full_t = build_kwargs['x_0_preproc_layer'](x_full_t, training=use_training)

    x_full_vs_t[i_t] = x_full_t
    x_full_t = transition_model(x_full_t, training=use_training)

    if i_t % 50 == 0:
        logging.info("At t={}".format(i_t))

fig = graphics.show_x_of_t_batch(np.expand_dims(x_hat_0, axis=0), 0)
fig.savefig(os.path.join(out_dir, 'X_hat_0.png'))
graphics.animate_x_vs_t_batch(x_full_vs_t, out_dir, dpi=200)
