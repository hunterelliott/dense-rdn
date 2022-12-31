"""
Script for testing ranges of parameters for variable-environment Gray-Scott models to e.g. find minimal lethal
variability
"""

import os
import logging

import numpy as np
import matplotlib.pyplot as plt

import utils
import graphics
from sampling import get_train_z
from analysis import get_compound_model_components, propagate_x
from models.transition import transition_model as get_transition_model
from therm import diffusion_entropy_rate_vs_time, gray_scott_reaction_entropy_rate_vs_time

in_model_dir = '/Users/hunterelliott/Iteration_B2/Gray_Scott_Models/v5_NESS/test_41/Nt1024'

feed_rate_var_amplitudes = np.linspace(0.0, .2, 20)
# feed_rate_var_amplitudes = [.25, .3, .4]
# feed_rate_var_period = 1e2  # Period of sinusoidal feed rate variation
# rand_feed_seed = None
transition_model_name = 'GaussianPulseFeedGrayScottTransitionLayer'
batch_size = 4
z_bits = 32
override_n_t = None
save_animations = False

# import random
# out_dir = '/Users/hunterelliott/TEMP/test_gs_sweep_ds_' + str(random.randint(1000, 9999))
out_dir = os.path.join(in_model_dir, 'analysis_0')


utils.posterity_log(out_dir, locals(), __file__)

model = utils.load_model(in_model_dir)
generator_model, transition_model_og, iteration_layer, encoder_model = get_compound_model_components(model)

# Get the (possibly fitted) gray scott parameters
init_decay_rate = transition_model_og.decay_rate.numpy()
init_feed_rate = transition_model_og.feed_rate.numpy()
init_diff_coef = transition_model_og.diff_coeffs.numpy()
d_t = transition_model_og.d_t

if override_n_t is None:
    n_t = iteration_layer.n_t
    logging.info("Using model-stored Nt value of {}".format(n_t))
else:
    n_t = override_n_t
    logging.info("Overriding Nt to user specified value of {}".format(n_t))

logging.info("Using parameters from loaded model:")
logging.info("decay rate of {} and base feed rate of {} and diffusion coef of {}".format(
    init_decay_rate, init_feed_rate, init_diff_coef))

x_0 = generator_model(get_train_z(generator_model.input_shape[1::], batch_size, z_bits), training=True)

var_vs_t = np.zeros((n_t+1, x_0.shape[-1], len(feed_rate_var_amplitudes)))
rxn_ds_vs_t = np.zeros((n_t+1, len(feed_rate_var_amplitudes)))
dif_ds_vs_t = np.zeros((n_t+1, len(feed_rate_var_amplitudes)))

for i, amp in enumerate(feed_rate_var_amplitudes):

    tic = utils.log_and_time("Running with variation amplitude {} ({} of {})".format(
        amp, i, len(feed_rate_var_amplitudes)))

    transition_model = get_transition_model(transition_model_name, d_t=d_t, init_diff_coef=init_diff_coef,
                                            init_feed_rate=init_feed_rate, init_decay_rate=init_decay_rate,
                                            pulse_period=8, pulse_amplitude=amp, pulse_sigma_xy=4)

    x_vs_t = propagate_x(x_0, transition_model, n_t)
    x_vs_t = transition_model.de_center_x(x_vs_t)

    # Spatial variance of each sample is averaged across the batch, preserving time and channel dimensions
    var_vs_t[:, :, i] = np.mean(np.var(x_vs_t, axis=(2, 3)), axis=1)
    logging.info("Variance at t=T : {}".format(var_vs_t[-1, :, i]))

    # Calculate mean dissipation rates over time for this amplitude
    dif_ds_vs_t[:, i], _ = diffusion_entropy_rate_vs_time(x_vs_t, transition_model.get_diff_coef(0))
    rxn_ds_vs_t[:, i], _ = gray_scott_reaction_entropy_rate_vs_time(x_vs_t, transition_model.get_decay_rate(0))

    logging.info("Diffusion dissipation rate at t=T : {}".format(dif_ds_vs_t[-1, i]))
    logging.info("Reaction dissipation rate at t=T : {}".format(rxn_ds_vs_t[-1, i]))

    utils.log_and_time(tic)

    if save_animations:

        amp_out_dir = os.path.join(out_dir, 'animations_amp_' + str(amp))
        os.makedirs(amp_out_dir)
        logging.info("Saving animation to {}".format(amp_out_dir))
        graphics.animate_x_vs_t_batch(transition_model.center_x(x_vs_t), amp_out_dir)


logging.info("Saving output to {}...".format(out_dir))

# Variance vs time, channel and perturbation amplitude figures

for i_chan in range(var_vs_t.shape[1]):

    fig = plt.figure()
    plt.imshow(var_vs_t[:, i_chan, :],
               extent=(feed_rate_var_amplitudes[0], feed_rate_var_amplitudes[-1], n_t, 0),
               aspect='auto', interpolation='none')
    plt.title(('Variance, channel {}'.format(i_chan)))
    plt.xlabel('Perturbation Amplitude')
    plt.ylabel('Timepoint')
    fig.savefig(os.path.join(out_dir, 'var_vs_time_and_amplitude_chan_' + str(i_chan) + '.png'), dpi=200)

fig = plt.figure()
plt.plot(feed_rate_var_amplitudes, np.transpose(var_vs_t[-1]))
plt.xlabel('Perturbation Amplitude')
plt.ylabel('Variance at t=T')
plt.legend(['Ch. ' + str(i_chan) for i_chan in range(var_vs_t.shape[1])])
plt.title('Terminal Variance')
fig.savefig(os.path.join(out_dir, 'var_T_vs_amplitude.png'))

np.save(os.path.join(out_dir, 'var_vs_time_and_amplitude'), var_vs_t)

# Dissipation rate vs time and perturbation amplitude figures

fig = plt.figure()
pos = plt.imshow(dif_ds_vs_t,
                 extent=(feed_rate_var_amplitudes[0], feed_rate_var_amplitudes[-1], n_t, 0),
                 aspect='auto', interpolation='none')
plt.title('Diffusion Dissipation Rate')
plt.xlabel('Perturbation Amplitude')
plt.ylabel('Timepoint')
fig.colorbar(pos)
fig.savefig(os.path.join(out_dir, 'dif_ds_vs_time_and_amplitude.png'), dpi=200)

fig = plt.figure()
pos = plt.imshow(rxn_ds_vs_t,
                 extent=(feed_rate_var_amplitudes[0], feed_rate_var_amplitudes[-1], n_t, 0),
                 aspect='auto', interpolation='none')
plt.title('Reaction Dissipation Rate')
plt.xlabel('Perturbation Amplitude')
plt.ylabel('Timepoint')
fig.colorbar(pos)
fig.savefig(os.path.join(out_dir, 'rxn_ds_vs_time_and_amplitude.png'), dpi=200)

# Terminal dissipation rate figure

fig = plt.figure()
plt.plot(feed_rate_var_amplitudes, np.stack([dif_ds_vs_t[-1], rxn_ds_vs_t[-1]], axis=-1))
plt.xlabel('Perturbation Amplitude')
plt.ylabel('Dissipation rate at t=T')
plt.legend(['Diffusion', 'Reaction'])
plt.title('Terminal Dissipation Rate')
fig.savefig(os.path.join(out_dir, 'ds_T_vs_amplitude.png'))

np.save(os.path.join(out_dir, 'ds_vs_time_and_amplitude'), [rxn_ds_vs_t, dif_ds_vs_t])

logging.info("Done!")
