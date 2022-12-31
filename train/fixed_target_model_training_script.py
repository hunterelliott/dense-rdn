"""
Script for configuring and then training "z-propagation" style models.
"""
import os
import math
import numpy as np
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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


##=========================================================================================================
## ==========!!!!! YOU NEED TO CHANGE THE X_OUTPUT of PROPAGATION_MODEL BACK BEFORE USING THIS SCRIPT AGAIN!!!=====##
##=========================================================================================================

x_shape = (256, 256, 5)
z_shape = (1, 1, 256)
z_bits = 64
n_t_iterable = [2**n for n in range(3, 12)]
# n_t_iterable = [1024]
x_output_period = lambda t: max(int(t / 32), 1)
# x_output_period = lambda t: 2

z_recon_loss_weight = 0 #1.0
x_loss_weight = 1.0
diffusion_entropy_loss_weight = 0  # -1.0  # Negative weights maximize entropy production. Zero disables.
reaction_entropy_loss_weight = 0  # -1.0
entropy_var_loss_weight = 0  # 1e1
entropy_var_mode = 'var'
dx_dt_loss_weight = 2e2

reset_models_each_n_t = False

has_therm_loss = diffusion_entropy_loss_weight != 0.0 or reaction_entropy_loss_weight != 0.0

# feed_rate = .03
# feed_conc = np.random.uniform(0, 1e-3, x_shape[-1])
# feed_conc = np.zeros(x_shape[-1])
# feed_conc[[0, 2]] = 1.0
# feed_conc[0::2] = 1.0
# feed_conc[0] = 1.0
fit_drive = True
# drive = FlowDriveLayer(init_feed_conc=feed_conc, init_flow_rate=feed_rate,
#                        fit_feed_conc=fit_drive, fit_flow_rate=fit_drive)
# drive = NoisyFlowDriveLayer(init_feed_conc=feed_conc, init_flow_rate=feed_rate, noise_amplitude=noise_amp)
# noise_amp = 10**-2

transition_kwargs = {'model_name': 'dense_kitchen_sink_crn',
                     # 'stoich_name': 'create_dense_2_species_mixed_order_reaction_network_stoichiometry',
                     'n_species': x_shape[-1],
                     'dx_dt_loss_weight': dx_dt_loss_weight,
                     'diffusion_entropy_loss_weight': diffusion_entropy_loss_weight,
                     'reaction_entropy_loss_weight': reaction_entropy_loss_weight,
                     'fit_diff_coeffs': True, 'fit_rate_const': True,
                     #'drive': drive,
                     'drive_kwargs': {'fit_flow_rate': fit_drive, 'fit_feed_conc': fit_drive},
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
    var_loss_func = lambda loss: dissipation_variation_loss(loss, mode=entropy_var_mode,
                                                            n_loss_groups=n_therm_losses,
                                                            weight=entropy_var_loss_weight)
    iteration_kwargs['meta_loss_func'] = var_loss_func

target_image = './assets/Ilya_Prigogine_1977c_compressed.jpg'
chan_map = [[0, 1]]


def target_corr_t(z, x):
    # Do this to avoid lambda and resulting log vomit
    return image_correlation_loss(z, x, utils.load_and_normalize_image(x_shape, target_image), chan_map, 1)


def target_corr_stochastic(z, x):
    # Do this to avoid lambda and resulting log vomit
    return image_correlation_loss(z, x, utils.load_and_normalize_image(x_shape, target_image), chan_map, 0)

def target_corr_stochastic_loss(z, x):
    # Minimizable loss form of correlation
    return 1 - image_correlation_loss(z, x, utils.load_and_normalize_image(x_shape, target_image), chan_map, 0)

def target_corr_stochastic_loss_mixed(z, x):
    # Minimizable loss form of correlation
    stoch_corr = image_correlation_loss(z, x, utils.load_and_normalize_image(x_shape, target_image), chan_map, 0)
    t_corr = image_correlation_loss(z, x, utils.load_and_normalize_image(x_shape, target_image), chan_map, 1)
    return 1 - (stoch_corr + t_corr) / 2

def target_corr_stochastic_loss_combined(z, x):
    # Minimizable loss form of
    corr = bounded_batch_correlation_loss(z, x, utils.load_and_normalize_image(x_shape, target_image), chan_map)
    return 1 - corr


compile_kwargs = {'loss_weights': {'state_encoder_model': z_recon_loss_weight,
                                   'state_encoder_model_1': 0.4 * z_recon_loss_weight,
                                   'state_encoder_model_2': z_recon_loss_weight,
                                   'x_output': x_loss_weight},
                  'x_loss': lambda z, x: image_similarity_loss(z, x, utils.load_and_normalize_image(x_shape, target_image), chan_map),
                  # 'x_loss': target_corr_stochastic_loss_combined,
                  'x_metrics': [target_corr_t, target_corr_stochastic]}

train_kwargs = {'batch_size': 2,
                'n_epochs': int(5e3),
                'lr': 1e-4,
                'z_bits': z_bits,
                'callback_kwargs': {'include_accuracy_saturation': z_recon_loss_weight != 0,
                                    'early_stopping_patience': 10000,
                                    'reduce_lr_patience': 5000,
                                    'lr_schedule': lambda e, l: step_lr_schedule(e, l, trigger_epochs=[(t+1)*2500 for t in range(3)], factor=0.5)},
                'compile_kwargs': compile_kwargs}


out_dir = '/media/hunter/fast storage/Training_Experiments/Iteration_C1/Dense_CRNs/v1_Fixed_Ilya/test_208'
# import random
# out_dir = '/Users/hunterelliott/TEMP/test_gs_crn_init/test_' + str(random.randint(1000, 9999))
# out_dir = '/home/hunter/Desktop/TEMP/test_combinedcorrloss/test_' + str(random.randint(1000, 9999))

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
