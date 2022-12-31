"""
Simple script to test out lorenz transition model
"""
import os
import logging

import numpy as np
import matplotlib.pyplot as plt

import graphics
from layers import SymbolicIterationLayer, IterationLayer
from models.mapping import generator_model
from models.transition import coupled_lorenz_transition_model
import utils
from sampling import get_train_z

from tensorflow.keras.layers import Input, GaussianNoise
from tensorflow.keras.models import Model

import tensorflow as tf


# ---- Config ---- #

# While each iteration may be slower this way, it saves enough time with graph tracing etc. to be overall faster
tf.config.experimental_run_functions_eagerly(True)
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

x_shape = (64, 64, 3)
z_shape = (1, 1, 16)
batch_size = 32
n_t = 1024
state_generator = 'simple_generator_model'
x_pad_generator = ((32, 16), (8, 128))
build_model = False # Whether to build a full keras model or just iterate in a for loop

import random
out_dir = '/Users/hunterelliott/TEMP/test_arbitrarypadG_' + str(random.randint(1000,9999))
utils.posterity_log(out_dir, locals(), __file__)


# ---- Iterate ---- #


G = generator_model(x_shape, z_shape, state_generator, x_pad=x_pad_generator)
# x_noised = GaussianNoise(0.5)(G.output, training=True)
# G = Model(inputs=G.inputs, outputs=x_noised)
C = coupled_lorenz_transition_model(x_shape)

if build_model:
    t = utils.log_and_time("Building model...")

    R = SymbolicIterationLayer(C, n_t)
    #R = IterationLayer(C, n_t)

    Z = Input(shape=z_shape)
    X_0 = G(Z, training=True)
    X_vs_t = tf.transpose(R(X_0), perm=[1, 0, 2, 3, 4])
    GL = Model(inputs=Z, outputs = X_vs_t)
    t_build = utils.log_and_time(t)
    logging.info("... or {} s/iteration.".format(str(t_build/n_t)))

    t = utils.log_and_time("Iterating...")
    x_vs_t = np.transpose(GL.predict(np.zeros(shape=(batch_size,) + z_shape)), [1, 0, 2, 3, 4])
    t_iterate = utils.log_and_time(t)
    logging.info("... or {} s/iteration.".format(str(t_iterate/n_t)))

else:
    logging.info("Iterating without building model...")
    z_0 = get_train_z(z_shape, batch_size)
    x_0 = G(z_0, training=True)
    x_vs_t = np.zeros((n_t + 1,) + x_0.shape)
    x_t = x_0

    t_start = utils.log_and_time("Iterating...")
    for t in range(n_t + 1):
        x_vs_t[t, :, :, :] = x_t
        x_t = C(x_t, training=True)

        if t % 50 == 0:
            logging.info("at t={}".format(t))

    t_iterate = utils.log_and_time(t_start)
    logging.info("... or {} s/iteration.".format(str(t_iterate/n_t)))

# ---- Save Results ---- #

logging.info("Saving output...")

mean_x_vs_t = np.mean(np.mean(np.mean(x_vs_t, axis=3), axis=2), axis=1)

fig = plt.figure()
plt.plot(mean_x_vs_t[:, 0], label='u')
plt.plot(mean_x_vs_t[:, 1], label='v')
plt.plot(mean_x_vs_t[:, 2], label='w')
plt.xlabel('Iterations')
plt.xlabel('Mean Value')
plt.legend()
fig.savefig(os.path.join(out_dir, 'mean_x_vs_t.png'))

fig = plt.figure()
example_trajectory = x_vs_t[:, int(batch_size/2), int(x_shape[0]/2), int(x_shape[1]/2), :]
ax = plt.axes(projection='3d')
ax.plot3D(example_trajectory[:, 0], example_trajectory[:, 1], example_trajectory[:, 2])
fig.savefig(os.path.join(out_dir, 'example_trajectory.png'))
#plt.show()

mean_at_end = np.mean(mean_x_vs_t[int(-n_t/10):, :], axis=0)
logging.info("Mean values over last 10% of iterations: {}".format(mean_at_end))

graphics.save_x_of_t_batch_snapshots(x_vs_t, [0, n_t], out_dir)

graphics.animate_x_vs_t_batch(x_vs_t, out_dir)

logging.info("Done.")

