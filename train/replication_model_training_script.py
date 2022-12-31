"""
Script for configuring and training "replication" models.
"""

import os
import math
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
from tensorflow.keras.layers import GaussianNoise

import graphics
import models
import utils
from analysis import get_compound_model_components, propagate_x
from training import train_iteration_model, step_lr_schedule
from sampling import get_train_z
from ops import (logits_loss, x_similarity_loss, accuracy, x_corr_loss, get_batch_trailing_fraction,
                 x_vs_t_similarity_loss)
from layers import RandomDaughterCenteringLayer

# ---- Config ---- #

x_shape = (32, 32, 5)
z_shape = (1, 1, 256)
z_bits = 24

x_buffer = 8

n_t = 256

batch_size = 1

use_x_similarity_loss = True
use_z_reconstruction_loss = False
diffusion_entropy_loss_weight = 0 # 5.0 / n_t
reaction_entropy_loss_weight = 0
dx_dt_loss_weight = 1.0
max_dx_dt = .3
fit_transition_model = True

build_kwargs = {'state_generator': 'rd_generator_model',
                'state_encoder': 'simple_encoder_model',
                'state_transition_model': 'dense_gs_crn', #'reversible_coupled_gray_scott_crn', #
                'batch_size': batch_size,
                'x_buffer': x_buffer,
                'x_yard': 0,  # x_shape[0] + 2*x_buffer,
                'x_outputs': use_x_similarity_loss,
                'z_outputs': use_z_reconstruction_loss,
                'granddaughter_fraction': 0,
                'n_t_x_loss': 1,
                'x_0_preproc_layer': None,  # GaussianNoise(0.0125),
                }

transition_kwargs = {'n_species': x_shape[-1],
                     'dx_dt_loss_weight': dx_dt_loss_weight, 'max_dx_dt': max_dx_dt,
                     'diffusion_entropy_loss_weight': diffusion_entropy_loss_weight,
                     'reaction_entropy_loss_weight': reaction_entropy_loss_weight,
                     'fit_diff_coeffs': fit_transition_model, 'fit_rate_const': fit_transition_model,
                     'drive_class': models.FlowDriveLayer,
                     'drive_kwargs': {'fit_flow_rate': fit_transition_model, 'fit_feed_conc': fit_transition_model},
                     'freeze_model': False
                     }
generator_kwargs = {}


x_metrics = [x_corr_loss]
z_metrics = [accuracy]

if build_kwargs['n_t_x_loss'] > 1:
    def x_ntx_corr(z_true, x_stack):
        return x_corr_loss(z_true, x_stack, t=build_kwargs['n_t_x_loss'])

    x_metrics = x_metrics + [x_ntx_corr]

if build_kwargs['granddaughter_fraction'] > 0:
    def x_gd_corr(z_true, x_stack):
        # Abuse scope a bit so tensorboard shows something intelligible instead of 'lambda'
        return x_corr_loss(*get_batch_trailing_fraction(z_true, x_stack, build_kwargs['granddaughter_fraction']))

    def x_gd_loss(z_true, x_stack):
        return x_vs_t_similarity_loss(*get_batch_trailing_fraction(z_true, x_stack, build_kwargs['granddaughter_fraction']))

    def gd_accuracy(z_true, z_pred):
        return accuracy(*get_batch_trailing_fraction(z_true, z_pred, build_kwargs['granddaughter_fraction']))

    x_metrics = x_metrics + [x_gd_corr, x_gd_loss]
    z_metrics = z_metrics + [gd_accuracy]

train_kwargs = {'batch_size': batch_size,
                'n_epochs': int(1e6),
                'lr':  1e-3,
                'clipnorm': .99,
                'z_bits': z_bits,
                'callback_kwargs': {'include_accuracy_saturation': False,
                                    'early_stopping_patience': 9000,
                                    'reduce_lr_patience': 3000,
                                    'lr_schedule': lambda e, l: step_lr_schedule(e, l, trigger_epochs=[100, 200, 300, 5000, 10000, 15000, 20000, 25000], factor=0.5),
                                    }
                }

compile_kwargs = {'loss_weights': {'state_encoder_model': 1.0,
                                   'state_encoder_model_1': 1.0,
                                   'state_encoder_model_2': 1.0,
                                   'x_output': 1e-1},
                  'x_loss': x_vs_t_similarity_loss,
                  'x_metrics': x_metrics,
                  'encoder_metrics': z_metrics}


# import random
# out_dir = '/Users/hunterelliott/TEMP/test_rdmodels/test_' + str(random.randint(1000, 9999))
# out_dir = '/home/hunter/Desktop/TEMP/test_gds/test_' + str(random.randint(1000, 9999))
out_dir = '/media/hunter/fast storage/Training_Experiments/Iteration_C1/Dense_CRNs/v4_Replication/test_88'

utils.posterity_log(out_dir, locals(), __file__)

# ---- Build Model ---- #

t_timer = utils.log_and_time("Building model...")

tf.random.get_global_generator()
#dist_strat = tf.distribute.MirroredStrategy()
#with dist_strat.scope():

model = models.iteration.replication_model(x_shape, z_shape, n_t, **build_kwargs,
                                           transition_kwargs=transition_kwargs, generator_kwargs=generator_kwargs)


utils.log_and_time(t_timer)

# ---- Train ---- #


train_iteration_model(model, out_dir, **train_kwargs, compile_kwargs=compile_kwargs)

# ---- Output ---- #


t_timer = utils.log_and_time("Saving output...")

utils.save_model(model, out_dir)

generator_model, transition_model, iteration_layer, encoder_model = get_compound_model_components(model)

# Run a batch through time...
z_0_test = get_train_z(z_shape, batch_size, entropy=z_bits)
x_0_test = generator_model(z_0_test, training=True)
if build_kwargs['x_0_preproc_layer'] is not None:
    x_0_test = build_kwargs['x_0_preproc_layer'](x_0_test, training=True)
x_vs_t = propagate_x(x_0_test, transition_model, n_t)

# ... and generate figures from it.
graphics.save_propagation_figures(x_vs_t, out_dir)
graphics.save_therm_figures(model, x_vs_t, out_dir)

# if build_kwargs['granddaughter_fraction'] > 0:
#     # TODO - check for better solution!
#     # Since tf.split seems to break when combined with the sub-batching of granddaughters and multi-GPU, we workaround
#     # like this.. copy weights to model created outside distribution scope
#     wts = model.get_weights()
#     model = models.iteration.replication_model(x_shape, z_shape, n_t, **build_kwargs,
#                                                transition_kwargs=transition_kwargs, generator_kwargs=generator_kwargs)
#     model.set_weights(wts)

if use_x_similarity_loss:
    x_stack_test = model(z_0_test, training=True)
    if use_z_reconstruction_loss:
        x_stack_test = x_stack_test[-1]
    graphics.save_replication_gallery(x_stack_test[0], out_dir)
    if build_kwargs['n_t_x_loss'] > 1:
        graphics.save_replication_gallery(x_stack_test[-1], out_dir,
                                          file_name='X012_t' + str(build_kwargs['n_t_x_loss']) + '_montage.png')

    x_stack_test = transition_model.de_center_x(x_stack_test)
    logging.info("Mean of X: {}, variance: {}".format(np.mean(x_stack_test[0, :, :, :, :, 0]),
                                                      np.var(x_stack_test[0, :, :, :, :, 0])))
    logging.info("Mean of X daughter 1: {}, variance: {}".format(np.mean(x_stack_test[0, :, :, :, :, 1]),
                                                                 np.var(x_stack_test[0, :, :, :, :, 1])))
    logging.info("Mean of X daughter 2: {}, variance: {}".format(np.mean(x_stack_test[0, :, :, :, :, 1]),
                                                                 np.var(x_stack_test[0, :, :, :, :, 1])))
    x_stack_test = transition_model.center_x(x_stack_test)

if build_kwargs['granddaughter_fraction'] > 0:
    logging.info("Saving granddaughter output...")
    out_dir_gd = os.path.join(out_dir, 'granddaughters')
    utils.prep_output_dir(out_dir_gd)

    if use_x_similarity_loss:
        # Save a gallery of granddaughters as well.
        n_gd = math.ceil(batch_size * build_kwargs['granddaughter_fraction'])
        graphics.save_replication_gallery(x_stack_test[0, -n_gd:, :, :, :, :], out_dir_gd)
    # And run a batch through time and visualize
    x_0_gd = RandomDaughterCenteringLayer(
        model.get_layer('random_daughter_centering_layer').x_daughter_offset)(x_vs_t[-1])
    if build_kwargs['x_0_preproc_layer'] is not None:
        x_0_gd = build_kwargs['x_0_preproc_layer'](x_0_gd, training=True)
    x_vs_t_gd = propagate_x(x_0_gd, transition_model, n_t)
    graphics.save_propagation_figures(x_vs_t_gd, out_dir_gd)
    graphics.save_therm_figures(model, x_vs_t_gd, out_dir_gd)

utils.log_and_time(t_timer)
