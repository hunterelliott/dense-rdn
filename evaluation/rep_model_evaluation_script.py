import os
import logging

import numpy as np

from tensorflow.keras.layers import ZeroPadding2D

import graphics
import models
import utils
from sampling import get_train_z
from ops import pearson_corr, asymmetric_corr, asymmetric_x_corr_loss
from analysis import propagate_x


model_dir = '/media/hunter/fast storage/Training_Experiments/Iteration_B1/Z_Replication_Models/v2_LocalMLPTransition/test_26'
batch_size = 48
z_bits = 24


out_dir = os.path.join(model_dir, '')
utils.posterity_log(out_dir, locals(), __file__)

os.environ['CUDA_VISIBLE_DEVICES'] = ''

model = utils.load_model(model_dir)

z_shape = model.input_shape[1::]
generator_model = [layer for layer in model.layers if layer.name == 'state_generator_model'][0]
x_shape = generator_model.output_shape[1::]
iteration_layer = [layer for layer in model.layers if layer.name == 'iteration_layer'][0]
n_t = iteration_layer.n_t
domain_shape = iteration_layer.input_shape[1::]
x_pad_generator = [[int((a-b)/2), int((a-b)/2)] for a, b in zip(domain_shape[0:2], x_shape[0:2])]


t_timer = utils.log_and_time("Running test batch...")

# Run a batch through time
z_0_test = get_train_z(z_shape, batch_size, entropy=z_bits)
x_0_test = ZeroPadding2D(x_pad_generator)(generator_model(z_0_test, training=True))
x_stack_test = model(z_0_test, training=True)

x_vs_t = propagate_x(x_0_test, iteration_layer.transition_model, n_t)

utils.log_and_time(t_timer)


t_timer = utils.log_and_time("Saving output...")

graphics.save_x_of_t_batch_snapshots(x_vs_t, [0, n_t], out_dir)
graphics.animate_x_vs_t_batch(x_vs_t, out_dir)
graphics.save_replication_gallery(x_stack_test, out_dir)

logging.info("Mean of X: {}, variance: {}".format(np.mean(x_stack_test[:, :, :, :, 0]),
                                                  np.var(x_stack_test[:, :, :, :, 0])))
logging.info("Mean of X daughter 1: {}, variance: {}".format(np.mean(x_stack_test[:, :, :, :, 1]),
                                                             np.var(x_stack_test[:, :, :, :, 1])))
logging.info("Mean of X daughter 2: {}, variance: {}".format(np.mean(x_stack_test[:, :, :, :, 1]),
                                                             np.var(x_stack_test[:, :, :, :, 1])))


r01 = pearson_corr(x_stack_test[:, :, :, :, 0], x_stack_test[:, :, :, :, 1])

loss = asymmetric_x_corr_loss(1.0, x_stack_test)

utils.log_and_time(t_timer)
