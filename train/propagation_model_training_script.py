"""
Script for configuring and then training "z-propagation" style models.
"""
import os
import math
import numpy as np
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import graphics
import models
import utils
from analysis import get_compound_model_components
from models.transition import transition_model as transition_model_builder
from models.drive import NoisyFlowDriveLayer, FlowDriveLayer
from layers import iteration_layer as iteration_layer_builder
from training import train_iteration_model, train_model_at_each_t, step_lr_schedule
from therm import dissipation_variation_loss
from ops import image_similarity_loss, image_correlation_loss, bounded_batch_correlation_loss


# ---- Config ---- #


x_shape = (64, 64, 5)
z_shape = (1, 1, 256)
z_bits = 256
n_t_iterable = [2**n for n in range(7, 12)]
# n_t_iterable = [1024]
x_output_period = lambda t: max(int(t / 32), 1)
# x_output_period = lambda t: 2

drive_noise_amp = 0.8
drive_noise_sigma = 3.0

z_recon_loss_weight = 1.0
x_loss_weight = 1.0
diffusion_entropy_loss_weight = .04  # Negative weights maximize entropy production. Zero disables.
reaction_entropy_loss_weight = 0
entropy_var_loss_weight = .04
entropy_var_mode = 'mad'
dx_dt_loss_weight = 1e3
max_dx_dt = .3 #.05 + drive_noise_amp

reset_models_each_n_t = True

has_therm_loss = diffusion_entropy_loss_weight != 0.0 or reaction_entropy_loss_weight != 0.0

# feed_conc = np.zeros(x_shape[-1])
# feed_conc[0] = 1.0
fit_drive = True

transition_kwargs = {'model_name': 'dense_gs_crn',
                     'n_species': x_shape[-1],
                     'dx_dt_loss_weight': dx_dt_loss_weight, 'max_dx_dt': max_dx_dt,
                     'diffusion_entropy_loss_weight': diffusion_entropy_loss_weight,
                     'reaction_entropy_loss_weight': reaction_entropy_loss_weight,
                     'fit_diff_coeffs': True, 'fit_rate_const': True,
                     'drive_class': NoisyFlowDriveLayer,
                     'drive_kwargs': {'fit_flow_rate': fit_drive, 'fit_feed_conc': fit_drive,
                                      'noise_amplitude': drive_noise_amp, 'filter_sigma': drive_noise_sigma},
                     'freeze_model': False
                     }


build_kwargs = {'state_transition_model': None,  # We fill this in later to allow kwargs rebuild + multi-gpu
                'state_encoder': 'simple_encoder_model',
                'state_generator': 'rd_generator_model',
                'x_pad_generator': 0,
                'x_pad_encoder': 0,
                'freeze_generator': False,
                'freeze_encoder': False,
                'x_outputs': x_loss_weight != 0 or has_therm_loss,
                'z_outputs': z_recon_loss_weight != 0,
                }

iteration_kwargs = {'layer_name': 'MixedStochasticOutputIterationLayer'}
if entropy_var_loss_weight != 0:
    assert has_therm_loss, "Must have at least one therm loss to have a variance loss!"
    n_therm_losses = int(diffusion_entropy_loss_weight != 0) + int(reaction_entropy_loss_weight != 0)
    var_loss_func = lambda loss, n_t: dissipation_variation_loss(loss, n_t, mode=entropy_var_mode,
                                                                 n_loss_groups=n_therm_losses,
                                                                 weight=entropy_var_loss_weight)
    iteration_kwargs['meta_loss_func'] = var_loss_func


compile_kwargs = {'loss_weights': {'state_encoder_model': z_recon_loss_weight,
                                   'state_encoder_model_1': 0.4 * z_recon_loss_weight,
                                   'state_encoder_model_2': z_recon_loss_weight,
                                   'x_output': x_loss_weight},
                  # 'x_loss': []
                  }

train_kwargs = {'batch_size': 2,
                'n_epochs': int(1e5),
                'lr': .25e-3,
                'z_bits': z_bits,
                'callback_kwargs': {'include_accuracy_saturation': False, # z_recon_loss_weight != 0,
                                    'early_stopping_patience': 6000,
                                    'reduce_lr_patience': 2000,
                                    'lr_schedule': lambda e, l: step_lr_schedule(e, l, trigger_epochs=[(t+1)*10000 for t in range(2)], factor=0.5)},
                'compile_kwargs': compile_kwargs}


out_dir = '/media/hunter/fast storage/Training_Experiments/Iteration_C1/Dense_CRNs/v3_DissInf/test_57'
# import random
# out_dir = '/Users/hunterelliott/TEMP/test_gs_crn_init/test_' + str(random.randint(1000, 9999))
# out_dir = '/home/hunter/Desktop/TEMP/test_dissinf_noise/test_' + str(random.randint(1000, 9999))

utils.posterity_log(out_dir, locals(), __file__)


# ---- Build Model ----#

# We have to create the global generator outside the distributed strategy scope
tf.random.get_global_generator().from_non_deterministic_state()

# This will use all GPUs if available and CPU if not.
# dist_strat = tf.distribute.MirroredStrategy()
# tf.config.run_functions_eagerly(True)

#with dist_strat.scope():


def model_builder(model, n_t):
    t_timer = utils.log_and_time("----Building Model----")
    if model is not None and not reset_models_each_n_t:
        logging.info("Re-using encoder, generator and transition model, rebuilding iterator...")
        generator_model, transition_model, _, encoder_model = get_compound_model_components(model)
        build_kwargs['state_generator'] = generator_model
        build_kwargs['state_encoder'] = encoder_model
        build_kwargs['state_transition_model'] = transition_model
    else:
        # (Re) build the transition model here using our kwargs, other models will be (re) built/(re)loaded below.
        build_kwargs['state_transition_model'] = transition_model_builder(**transition_kwargs)
        logging.info("(Re)building all models...")

    if has_therm_loss:
        # Make sure that the entropy losses are independent of timescale.
        new_val = diffusion_entropy_loss_weight / (n_t * build_kwargs['state_transition_model'].d_t)
        build_kwargs['state_transition_model'].diffusion_entropy_loss_weight = new_val
        logging.info("Rescaling diffusion_entropy_loss_weight to {}".format(new_val))

        new_val = reaction_entropy_loss_weight / (n_t * build_kwargs['state_transition_model'].d_t)
        build_kwargs['state_transition_model'].reaction_entropy_loss_weight = new_val
        logging.info("Rescaling reaction_entropy_loss_weight to {}".format(new_val))

    if dx_dt_loss_weight != 0:
        # Make sure that step size penalty is independent of timescale
        new_val = dx_dt_loss_weight / (n_t * build_kwargs['state_transition_model'].d_t)
        build_kwargs['state_transition_model'].dx_dt_loss_weight = new_val
        logging.info("Rescaling dx/dt penalty weight to {}".format(new_val))


    # We re-build the iteration model every time since n_t changes and it has no weights
    iteration_kwargs['x_output_period'] = x_output_period(n_t)
    build_kwargs['iteration_layer'] = iteration_layer_builder(transition_model=build_kwargs['state_transition_model'],
                                                              n_t=n_t, **iteration_kwargs)

    model = models.iteration.propagation_model(x_shape, z_shape, n_t, batch_size=train_kwargs['batch_size'],
                                               x_output_period=x_output_period(n_t), **build_kwargs)
    utils.log_and_time(t_timer)
    return model


def model_saver(model, out_dir):

    graphics.save_propagation_model_and_figures(model, out_dir, z_bits=z_bits)


# ---- Train ---- #

logging.info("----Starting Training Loop----")
train_model_at_each_t(model_builder, model_saver, train_iteration_model,
                      n_t_iterable, out_dir, **train_kwargs)
logging.info("----Training Loop Completed!----")
