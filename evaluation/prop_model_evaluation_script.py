
import os
import logging
import math

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import numpy as np

import graphics
import utils
import analysis
import models

import tensorflow as tf


"""
A simple script for figure generation etc. on an already-trained model.
"""


z_bits = 16
n_t_override = 10240
d_t_override = None
model_dir = '/Users/hunterelliott/Iteration_C1/Dense_CRNs/v3_DissInf/test_53/Nt128'
use_train_time = True


out_name = 'evaluation'
if n_t_override is not None:
    out_name = out_name + '_' + 'Nt' + str(n_t_override)
if d_t_override is not None:
    out_name = out_name + '_' + 'dt' + str(d_t_override)

out_dir = os.path.join(model_dir, out_name)

utils.posterity_log(out_dir, locals(), __file__)
logging.info("Running evaluation on model from {}".format(model_dir))


model = utils.load_model(model_dir)

x_vs_t, z_vs_t, z_acc_vs_t, z_dist_vs_t = analysis.propagate_x_and_z(model, n_t=n_t_override, d_t=d_t_override,
                                                                     z_bits=z_bits,
                                                                     use_train_time=use_train_time, batch_size=9)
n_t = x_vs_t.shape[0]-1

logging.info("Generating figures, saving to {}".format(out_dir))
if any(['encoder' in output_name for output_name in model.output_names]) and x_vs_t.shape[0] > 1:
    graphics.save_z_propagation_figures(z_dist_vs_t, z_acc_vs_t, out_dir)

graphics.save_therm_figures(model, x_vs_t, out_dir)
graphics.save_propagation_figures(x_vs_t, out_dir, contrast=-10)


transition_model = analysis.get_compound_model_components(model)[1]
if 'gray_scott' in transition_model.name:
    # TODO - handle new drive layers...
    batch_size = x_vs_t.shape[1]
    f_vs_t = np.zeros(x_vs_t.shape[0:-1])
    t_start = utils.log_and_time("Getting feed rates vs time...")
    for i_t in range(n_t+1):
        f_vs_t[i_t, :] = transition_model.get_feed_rate(np.ones(batch_size)*i_t, seed=42)
        if i_t % 50 == 0:
            logging.info("...at t={}".format(i_t))
    utils.log_and_time(t_start)
    f_vs_t = np.expand_dims(f_vs_t, axis=-1)
    feed_out_dir = os.path.join(out_dir, 'feed_rate')
    os.makedirs(feed_out_dir)
    graphics.save_x_of_t_batch_snapshots(f_vs_t, [0, 10, int(n_t/4), int(n_t/2), int(3*n_t/4), n_t], feed_out_dir)
    graphics.animate_x_vs_t_batch(f_vs_t, feed_out_dir, x_range=(np.amin(f_vs_t), np.amax(f_vs_t)))
