import logging
import math
import numpy as np
from models.transition import BrusselatorTransitionLayer

import graphics
from analysis import propagate_x
import utils

w = 128
x_shape = (w, w, 2)
batch_size = 9
n_t = int(1e4)
subsample_anim = 10

Dx = 1.0
Dy = 8.0
A = 4.5
# Use mu control parameter from Pena & Perez-Garcia PRL 2001
mu = .08
eta = math.sqrt(1 / Dy) # Assumes Dx = 1
Bcrit = (1 + A*eta) ** 2
B = Bcrit*(mu + 1)

d_t = .01

transition_layer = BrusselatorTransitionLayer(A, B, Dx, Dy, d_t=d_t)

import random
out_dir = '/Users/hunterelliott/TEMP/test_brusselator_compare_crn/test_A' + str(A) + '_B' + str(B) + '_mu' + str(mu) + '_dx_' + str(Dx) + '_' + '_dy_' + str(Dy) + str(random.randint(1000, 9999))
utils.posterity_log(out_dir, locals(), __file__)


delta = .1
x_0 = np.zeros((batch_size,) + x_shape, dtype=np.float32)
x_0[:, :, :, 0] = A + np.random.uniform(-delta, delta, x_0.shape[0:-1])
x_0[:, :, :, 1] = B/A + np.random.uniform(-delta, delta, x_0.shape[0:-1])
x_0 = (x_0 * 2) - 1


t_start = utils.log_and_time("Starting iteration...")
x_vs_t = propagate_x(x_0, transition_layer, n_t)
utils.log_and_time(t_start)

logging.info("Saving output...")

graphics.save_x_of_t_batch_snapshots(x_vs_t, [0, 10, int(n_t/4), int(n_t/2), int(3*n_t/4), n_t], out_dir)
graphics.animate_x_vs_t_batch(x_vs_t[0:n_t+1:subsample_anim], out_dir)
