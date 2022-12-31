import os
import logging
import math

import numpy as np
from tensorflow.keras.layers import Lambda

from matplotlib import pyplot as plt

from models.transition import GrayScottTransitionLayer, VariableFeedGrayScottTransitionLayer, \
    SinusoidalFeedGrayScottTransitionLayer, RandomFeedGrayScottTransitionLayer, \
    GaussianGridFeedGrayScottTransitionLayer, GaussianPulseFeedGrayScottTransitionLayer
from models.mapping import simple_generator_model
from sampling import get_train_z
import graphics
from analysis import propagate_x
import utils


w = 128
x_shape = (w, w, 2)
batch_size = 9
n_t = 5000
subsample_anim = 50
initial_cond = 'Pearson'
F = .03
# #F = Lambda(lambda t: .04)
# def t_schedule(t):
#     base_feed = .04
#     t_start = 5e3
#     if t < t_start:
#         return base_feed
#     else:
#         return base_feed + .016*math.sin((t-t_start) / 1e2)
# F = t_schedule
k = 0.06 #.065
d = (.2, .1)

#transition_layer = GaussianPulseFeedGrayScottTransitionLayer(init_feed_rate=F, init_decay_rate=k, pulse_sigma_xy=3, pulse_period=16, pulse_amplitude=0.0125, pulse_sigma_t=64, max_n_t=1536, constant_baseline=True)

transition_layer = GrayScottTransitionLayer(init_feed_rate=F, init_decay_rate=k, init_diff_coef=d)

import random
out_dir = '/Users/hunterelliott/TEMP/test_GS_compare_coupled_crn/test_f' +str(F) + '_k' + str(k) + '_d_' + str(d) + '_' + str(random.randint(1000, 9999))
# out_dir = '/Users/hunterelliott/Iteration_B2/Gray_Scott_Models/v4_ScaleUp/test_60/Nt1024/pearson_init_matched_params'
utils.posterity_log(out_dir, locals(), __file__)


if initial_cond == 'Pearson':
    g_w = int(w/4)
    x_0 = np.zeros((batch_size,) + x_shape, dtype=np.float32)
    # Reproducing the initial conditions given in the Pearson paper
    x_0[:, :, :, 0] = 1.0
    x_0[:, int(w/2)-g_w:int(w/2)+g_w,  int(w/2)-g_w:int(w/2)+g_w, 0] = 0.5
    x_0[:, int(w/2)-g_w:int(w/2)+g_w,  int(w/2)-g_w:int(w/2)+g_w, 1] = 0.25
    x_0 = x_0 * np.random.uniform(.99, 1.01, x_0.shape)
    # Compensate for the expected range
    x_0 = (x_0 * 2) - 1

elif initial_cond == 'Random':
    x_0 = np.zeros((batch_size,) + x_shape, dtype=np.float32)
    x_0[:, :, :, 0] = np.random.uniform(.4, .6, (batch_size,) + x_shape[0:2]).astype(np.float32)
    x_0[:, :, :, 0] = np.random.uniform(.15, .35, (batch_size,) + x_shape[0:2]).astype(np.float32)
    # Compensate for the expected range
    x_0 = (x_0 * 2) - 1
elif initial_cond == 'Generator':
    z_shape = (1, 1, 256)
    G = simple_generator_model(x_shape, z_shape)
    x_0 = G(get_train_z(z_shape, batch_size))

#C = transition_layer(init_feed_rate=F, init_decay_rate=k, amplitude=.005)
C = transition_layer# (init_feed_rate=F, init_decay_rate=k, max_n_t=2048, pulse_period=128) #, n_grid_divisions=8)

t_start = utils.log_and_time("Starting iteration...")
x_vs_t = propagate_x(x_0, C, n_t)
utils.log_and_time(t_start)

logging.info("Saving output...")

graphics.save_x_of_t_batch_snapshots(x_vs_t, [0, 10, int(n_t/4), int(n_t/2), int(3*n_t/4), n_t], out_dir)
graphics.animate_x_vs_t_batch(x_vs_t[0:n_t+1:subsample_anim], out_dir)

f_vs_t = np.zeros((n_t+1, batch_size) + x_shape[0:-1])
t_start = utils.log_and_time("Getting feed rates vs time...")
for i_t in range(n_t+1):
    f_vs_t[i_t, :] = C.get_feed_rate(np.ones(batch_size)*i_t, seed=42)
    if i_t % 50 == 0:
        logging.info("...at t={}".format(i_t))
utils.log_and_time(t_start)
f_vs_t = np.expand_dims(f_vs_t, axis=-1)
feed_out_dir = os.path.join(out_dir, 'feed_rate')
os.makedirs(feed_out_dir)
graphics.save_x_of_t_batch_snapshots(f_vs_t, [0, 10, int(n_t/4), int(n_t/2), int(3*n_t/4), n_t], feed_out_dir)
graphics.animate_x_vs_t_batch(f_vs_t[0:n_t+1:subsample_anim], feed_out_dir, x_range=(np.amin(f_vs_t), np.amax(f_vs_t)))

j=1

