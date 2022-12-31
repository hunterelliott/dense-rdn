import logging
import math

import numpy as np
import tensorflow as tf

from ops import laplacian_kernel, pad_toroidal
import utils, graphics
from models.drive import FlowDriveLayer, ConstantSynthesisDriveLayer, NoisyFlowDriveLayer
from models.transition import ReactionNetworkTransitionLayer
from analysis import propagate_x


# Setup the matrices that describe the reaction system.
system_name = 'coupled_gray_scott'

if system_name == 'gray_scott':
    # n_species x n_reactions matrix of reactant and product stoichiometry
    reactants = [[1, 0, 0],
                 [2, 1, 3]]

    products =  [[0, 0, 1],
                 [3, 0, 2]]

    # The rate constants for each reaction
    k_rxn = [1.0, .06, .00001]


    # We specify feed concentrations for each species
    feed_conc = tf.convert_to_tensor([1.0, 0.0], dtype=tf.float32)
    # And a global feed rate.
    feed_rate = 0.03

    # Diffusion coefficients
    diff_coeff = [.2, .1]

    drive_layer = FlowDriveLayer(init_feed_conc=feed_conc, init_flow_rate=feed_rate)

    d_t = 1

elif system_name == 'brusselator':
    # B gets folded into a X->Y conversion rate constant
    reactants = [[2, 1, 1],
                 [1, 0, 0]]

    products = [[3, 0, 0],
                [0, 1, 0]]

    Dx = 1.0
    Dy = 8.0
    A = 4.5
    # Use mu control parameter from Pena & Perez-Garcia PRL 2001
    mu = .08
    eta = math.sqrt(1 / Dy)  # Assumes Dx = 1
    Bcrit = (1 + A * eta) ** 2
    B = Bcrit * (mu + 1)
    k_rxn = [1.0, B, 1.0]

    diff_coeff = [Dx, Dy]

    synth_rate = tf.convert_to_tensor([A, 0.0], dtype=tf.float32)

    drive_layer = ConstantSynthesisDriveLayer(synth_rate)

    d_t = .01

elif system_name == '3_species_gray_scott':
    # n_species x n_reactions matrix of reactant and product stoichiometry
    reactants = [[1, 0, 0, 0],
                 [2, 1, 1, 0],
                 [0, 0, 2, 1]]

    products = [[0, 0, 0, 0],
                [3, 0, 0, 0],
                [0, 0, 3, 0]]

    # The rate constants for each reaction
    k_rxn = [1, .06, .5, .012]

    # We specify feed concentrations for each species
    feed_conc = tf.convert_to_tensor([1.0, 0.03, 0.01], dtype=tf.float32)
    # And a global feed rate.
    feed_rate = 0.03

    # Diffusion coefficients
    diff_coeff = [.2, .1, .03]

    drive_layer = FlowDriveLayer(init_feed_conc=feed_conc, init_flow_rate=feed_rate)

    d_t = 1

elif system_name == 'coupled_gray_scott':
    # n_species x n_reactions matrix of reactant and product stoichiometry
    reactants = [[1, 0, 0, 0, 0],
                 [2, 1, 0, 0, 1],
                 [0, 0, 1, 0, 0],
                 [0, 0, 2, 1, 0]]

    products = [[0, 0, 0, 0, 0],
                [3, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 3, 0, 1]]

    # The rate constants for each reaction
    k_rxn = [1, .06, 1.0, .056, 0.0015]

    # We specify feed concentrations for each species
    feed_conc = tf.convert_to_tensor([1.0, 0.0, 1.0, 0.0], dtype=tf.float32)
    # And a global feed rate.
    feed_rate = 0.03

    # Diffusion coefficients
    diff_coeff = [.2, .05, .16, .1]

    #drive_layer = FlowDriveLayer(init_feed_conc=feed_conc, init_flow_rate=feed_rate)
    drive_layer = NoisyFlowDriveLayer(init_feed_conc=feed_conc, init_flow_rate=feed_rate,
                                      noise_amplitude=.15, filter_sigma=3)

    d_t = 1

reactants = np.array(reactants)
products = np.array(products)

w = 128
n_species = reactants.shape[0]
x_shape = (w, w, n_species)
batch_size = 2
n_t = int(1e4)
subsample_anim = 10
init_cond = 'coupled_pearson'


# --- Init ---- #

import random
out_dir = '/Users/hunterelliott/TEMP/coupled_gs_examples/test_' + str(random.randint(1000, 9999))
#out_dir = '/home/hunter/Desktop/TEMP/test_rxn_net_4species/test_' + str(random.randint(1000, 9999))
utils.posterity_log(out_dir, locals(), __file__)

crn_model = ReactionNetworkTransitionLayer(reactants=reactants, products=products, init_rate_const=k_rxn,
                                           init_diff_coeffs=diff_coeff, drive=drive_layer, d_t=d_t)

if init_cond == 'pearson':
    # Initial X with Pearson's initial conditions.
    g_w = 32
    x_0 = np.zeros((batch_size,) + x_shape, dtype=np.float32)
    x_0[:, :, :, 0] = 1.0
    x_0[:, int(w/2)-g_w:int(w/2)+g_w,  int(w/2)-g_w:int(w/2)+g_w, 0] = 0.5
    x_0[:, int(w/2)-g_w:int(w/2)+g_w,  int(w/2)-g_w:int(w/2)+g_w, 1] = 0.25
    x_0 = x_0 * np.random.uniform(.99, 1.01, x_0.shape)

elif init_cond == 'pearson3':
    # Initial X with Pearson's initial conditions.
    g_w = int(w / 4)
    x_0 = np.zeros((batch_size,) + x_shape, dtype=np.float32)
    x_0[:, :, :, 0] = 1.0
    x_0[:, int(w/2)-g_w:int(w/2)+g_w,  int(w/2)-g_w:int(w/2)+g_w, 0] = 0.5
    x_0[:, int(w / 2) - g_w:int(w / 2) + g_w, int(w / 2) - g_w:int(w / 2) + g_w, 1] = 0.25
    x_0[:, int(w / 2) - g_w:int(w / 2) + g_w, int(w / 2) - g_w:int(w / 2) + g_w, 2] = 0.25
    x_0 = x_0 * np.random.uniform(.99, 1.01, x_0.shape)

elif init_cond == 'coupled_pearson':
    # Initial X with modified form of Pearson's initial conditions.
    g_w = int(w / 4)
    h_w = int(w / 8)
    x_0 = np.zeros((batch_size,) + x_shape, dtype=np.float32)
    x_0[:, :, :, 0] = 1.0
    x_0[:, :, :, 2] = 1.0
    x_0[:, int(w/2)-g_w:int(w/2)+g_w,  int(w/2)-g_w:int(w/2)+g_w, 0] = 0.5
    x_0[:, int(w / 2) - g_w:int(w / 2) + g_w, int(w / 2) - g_w:int(w / 2) + g_w, 1] = 0.25
    x_0[:, int(w/2)-h_w:int(w/2)+h_w,  int(w/2)-h_w:int(w/2)+h_w, 2] = 0.5
    x_0[:, int(w / 2) - h_w:int(w / 2) + h_w, int(w / 2) - h_w:int(w / 2) + h_w, 3] = 0.25
    x_0 = x_0 * np.random.uniform(.99, 1.01, x_0.shape)

elif init_cond == 'random':
    x_0 = np.random.uniform(.9, 1.1, (batch_size,) + x_shape).astype(np.float32)
elif init_cond == 'brusselator':
    delta = .1
    x_0 = np.zeros((batch_size,) + x_shape, dtype=np.float32)
    x_0[:, :, :, 0] = A + np.random.uniform(-delta, delta, x_0.shape[0:-1])
    x_0[:, :, :, 1] = B / A + np.random.uniform(-delta, delta, x_0.shape[0:-1])


# Compensate for the expected range
x_0 = (x_0 * 2) - 1
x_0 = tf.convert_to_tensor(x_0, dtype=tf.float32)

# ----- Integration ------ #

t_start = utils.log_and_time("Starting iteration...")

x_vs_t = propagate_x(x_0, crn_model, n_t)

utils.log_and_time(t_start)

logging.info("Saving output...")

if n_species > 3:
    x_vs_t = graphics.project_channels_to_rgb(x_vs_t)
graphics.save_x_of_t_batch_snapshots(x_vs_t, [0, 10, int(n_t/4), int(n_t/2), int(3*n_t/4), n_t], out_dir)
graphics.animate_x_vs_t_batch(x_vs_t[0:n_t+1:subsample_anim], out_dir)
