"""
Module with thermodynamic functions / utils / losses etc.
"""
import logging

import numpy as np
from scipy.constants import Boltzmann
import tensorflow as tf
from tensorflow_addons.image import gaussian_filter2d

from chem import get_reaction_network_rates, find_matching_reactions

k_b = tf.cast(Boltzmann, tf.float32)


def diffusion_entropy_rate(c, diff_coef, epsilon=1e-1, pre_filter=False):
    """
    Calculates the rate of entropy production due to diffusion for the input chemical concentration state.

    The formula is due to [1] though we have included the missing factor of k_b, divided by area to make the rate a
    spatial average rather than sum, and also added a numerical stabilizer.

    Args:
        c: (batch, height, width, n_species) matrix of local chemical concentrations (e.g. in mol/L).
        diff_coef: (n_species,) tuple of diffusion coefficients for each chemical species (m^2 / s)
        epsilon: A numerical stabilizer to prevent divide-by-zero.
        pre_filter: If true, concentrations are gaussian pre-filtered to improve reliability of gradient estimates at
            the expense of high-frequency concentration variation.

    Returns:
        delta_s: (batch,) vector of the per-sample rate of entropy production
            (in (J * C) / (K * s) where C is the concentration units e.g. mol/L)

    [1] Mahara et al., Chaos 15, 047508 (2005)
    """
    c = tf.convert_to_tensor(c, dtype=tf.float32)
    if pre_filter:
        # Improve gradient estimates by pre-filtering the image.
        c = gaussian_filter2d(c, 3, 1.0)

    dx, dy = tf.image.image_gradients(c)
    diff_coef = tf.expand_dims(tf.cast(diff_coef, tf.float32), 0)

    return tf.reduce_sum(tf.math.reduce_mean((dx ** 2 + dy ** 2) / (c + epsilon), axis=(1, 2)) * diff_coef, axis=1) * k_b


def diffusion_entropy_rate_vs_time(c_vs_t, diff_coef):
    """
    Calculates the entropy production rate due to diffusion over time in the input concentration timeseries

    Args:
        c_vs_t: (time, batch_size, height, width, n_species) matrix of local chemical species concentrations over time.
        diff_coef: (n_species) vector of diffusion constants of each species.

    Returns:
        mean_vs_time, std_vs_time: (n_timepoints) vectors of the batch mean and standard deviation entropy production
            rates

    """
    n_t = c_vs_t.shape[0]
    batch_size = c_vs_t.shape[1]
    ds_vs_t = np.zeros((batch_size, n_t))
    for t in range(c_vs_t.shape[0]):
        ds_vs_t[:, t] = diffusion_entropy_rate(c_vs_t[t], diff_coef)

    return np.mean(ds_vs_t, axis=0), np.std(ds_vs_t, axis=0)


def reaction_entropy_rate(v_fwd, v_rev, epsilon=1e-5):
    """
    Calculates the entropy production rate due to chemical reactions with local reaction rates described by the input
    matrices [1]. Note that the definition of 'forward' vs 'reverse' is arbitrary, it is only important that the two
    reactions be their own inverse (reactants of one are products of the other and vice versa).

    Args:
        v_fwd: (batch, height, width, n_reactions) array with local 'forward' reaction rates for each 'forward' reaction,
            in units of mol / s.
        v_rev: (batch, height, width, n_reactions) array with local 'reverse' reaction rates for each 'forward' reaction,
            in units of mol / s, in the same order as v_fwd e.g. v_rev[:,:,:,i] is the reverse of v_fwd[:,:,:,i]
        epsilon: Numerical stabilizer to prevent log(0) and divide by 0.

    Returns:
        delta_s: a (batch,) vector of the per-sample reaction entropy rates, averaged over the domain, in (J*C)/(K*s)

    [1] Kondepudi and Prigogine, Modern Thermodynamics 2nd ed, 9.5.11
    """
    assert len(v_fwd.shape) == 4 and v_fwd.shape[-1] > 0, "Rate arrays must be 4D and non-empty!"
    assert v_fwd.shape[-1] == v_rev.shape[-1], "Rate arrays must be 4D and non-empty!"

    # Avoid singularities due to low rates
    v_fwd = tf.where(v_fwd < epsilon, epsilon, v_fwd)
    v_rev = tf.where(v_rev < epsilon, epsilon, v_rev)

    # Calculate local entropy rate, gives (batch, height, width, n_matched_reactions)
    local_rate = (v_fwd - v_rev) * tf.math.log(v_fwd / v_rev)
    # Average over the spatial domain so we are independent of domain size
    # Giving a (batch, n_matched_reactions) matrix of per-reaction average entropy rates
    mean_rate = tf.reduce_mean(local_rate, axis=(1, 2))

    # Finally, sum over the reactions
    return tf.reduce_sum(mean_rate, axis=1) * k_b


def reaction_network_entropy_rate_vs_time(c_vs_t, reactants, products, rate_const):
    """
        Calculates the entropy production rate due to reactions over time in the input chemical reaction network
         concentration timeseries.

        Args:
            c_vs_t: (time, batch_size, height, width, n_species) matrix of local chemical species concentrations over
                time.
            reactants: (n_species, n_reactions) matrix of reactant stoichiometries
            products: (n_species, n_reactions) matrix of product stoichiometries
            rate_const: (n_reactions) vector of reaction rate constants


        Returns:
            mean_vs_time, std_vs_time: (n_timepoints) vectors of the batch mean and standard deviation entropy
                production rates

    """
    reaction_matches = find_matching_reactions(reactants, products)
    assert reaction_matches.size == reactants.shape[1], "The input reaction network must be fully reversible!"

    n_t = c_vs_t.shape[0]
    batch_size = c_vs_t.shape[1]
    ds_vs_t = np.zeros((batch_size, n_t))
    for t in range(c_vs_t.shape[0]):
        rates = get_reaction_network_rates(c_vs_t[t], reactants, rate_const)
        ds_vs_t[:, t] = reaction_entropy_rate(tf.gather(rates, reaction_matches[:, 0], axis=3),
                                              tf.gather(rates, reaction_matches[:, 1], axis=3))

    return np.mean(ds_vs_t, axis=0), np.std(ds_vs_t, axis=0)


def irreversible_reaction_entropy_rate(v_fwd, v_rev=1e-5, epsilon=1e-3):
    """
    Calculates the entropy production rate due to chemical reactions with local reaction rates described by the input
    matrices.

    Formula is due to [1] (same as in Kondepudi & Prigogine text) though we use irreversible reaction models and assume
    a small constant reverse reaction rate for computational simplicity. This makes the entropy rate defined and so
    long as the reverse rate is small (say < 1e-3) they confirm it does not affect the overall system behavior.

    Args:
        v_fwd: A list of (batch, height, width) matrices with local forward reaction rates for each reaction, in units
            of C / s where C is the concentration unit.
        v_rev: A scalar constant specifying the (small) fixed reaction rate.
        epsilon: Numerical stabilizer to prevent log(0).

    Returns:
        delta_s: a (batch,) vector of the per-sample reaction entropy rates, averaged over the domain, in (J*C)/(K*s)

    [1] Mahara et al., Entropy 12, 2436 (2010)
    """

    ds_per_rxn = []
    for v in v_fwd:
        v = tf.convert_to_tensor(v, dtype=tf.float32)
        ds_per_rxn.append(tf.reduce_mean(tf.math.log((v + epsilon) / v_rev) * (v - v_rev), axis=(1, 2)))

    return tf.math.accumulate_n(ds_per_rxn) * k_b


def gray_scott_reaction_entropy_rate(c, decay_rate):
    """
    A wrapper for calculating the reaction entropy production rate for the gray-scott reaction system given only
    concentrations.
    Args:
        c: (batch, height, width, 2) local concentration matrix
        decay_rate:

    Returns:
        delta_s: a (batch,) vector of the per-sample reaction entropy rates, averaged over the domain.

    """

    # Reaction rate of the autocatalytic reaction
    v_1 = c[:, :, :, 0] * c[:, :, :, 1] ** 2

    # Reaction rate of the catalyst decay reaction
    v_2 = decay_rate * c[:, :, :, 1]

    return irreversible_reaction_entropy_rate([v_1, v_2])


def gray_scott_reaction_entropy_rate_vs_time(c_vs_t, decay_rate):
    """
    Calculates the entropy production rate due to reactions over time in the input gray-scott model concentration
    timeseries.

    Args:
        c_vs_t: (time, batch_size, height, width, 2) matrix of local chemical species concentrations over time.
        decay_rate: The reaction rate for the decay process.

    Returns:
        mean_vs_time, std_vs_time: (n_timepoints) vectors of the batch mean and standard deviation entropy production
            rates

    """
    n_t = c_vs_t.shape[0]
    batch_size = c_vs_t.shape[1]
    ds_vs_t = np.zeros((batch_size, n_t))
    for t in range(c_vs_t.shape[0]):
        ds_vs_t[:, t] = gray_scott_reaction_entropy_rate(c_vs_t[t], decay_rate)

    return np.mean(ds_vs_t, axis=0), np.std(ds_vs_t, axis=0)


def mean_concentration_delta(c0, c1):
    """
    Calculates the change in mean concentration for each species in the input concentration matrices.
    Args:
        c0: (batch, height, width, n_species) matrix of local concentrations at t=0
        c1: (batch, height, width, n_species) matrix of local concentrations at t=1

    Returns:
        delta_c: (batch, n_species) matrix of differences in mean concentration.
    """

    # Get the per-channel means for each sample at t=0 and t=1
    c_mean_0 = tf.reduce_mean(c0, axis=(1, 2))
    c_mean_1 = tf.reduce_mean(c1, axis=(1, 2))

    return c_mean_1 - c_mean_0


def dissipation_variation_loss(losses, n_t, mode='var', n_loss_groups=1, weight=1.0, is_scaled_by_nt=True):
    """
    A "metaloss" function which penalizes the change in dissipation rates over time,
    to be used as an IterationLayer 'meta_loss_func'.

    Args:
        losses: loss list from transition model
        n_t: Number of timepoints
        mode: Which form of loss to use.
        n_loss_groups: Number of different losses applied to each timepoint, needed for eager mode.
        weight: rescale loss by this factor.
        is_scaled_by_nt: If the losses were already scaled by n_t then that will be accounted for here

    Returns:
        var_mean: the average of the variance of reaction and diffusion entropy rates

    """
    if is_scaled_by_nt:
        # If loss was divided by n_t we need to multiply by n_t before e.g. calculating variance and then dividing again
        scale = n_t
    else:
        scale = 1.0

    var_mean = 0.0
    for (i_group, group) in enumerate(['diffusion_entropy_rate_loss', 'reaction_entropy_rate_loss']):
        if tf.executing_eagerly():
            logging.warning("In eager mode this loss assumes there are ONLY dissipation losses on the transition models!!")
            if i_group < n_loss_groups:
                # To work in eager mode we have to just rely on the indices being in order since op names
                # aren't specified. What a mess.
                curr_losses = losses[i_group::n_loss_groups]
            else:
                curr_losses = []
        else:
            curr_losses = [l for l in losses if hasattr(l, 'name') and group in l.name]

        if len(curr_losses) > 0:
            curr_losses = tf.stack(curr_losses) * scale

            logging.info("Applying {} loss to {} transition model losses from group {}".format(
                mode, len(curr_losses), group))
            if mode == 'var':
                # Variance for this group
                var_mean = var_mean + tf.math.reduce_variance(curr_losses)
            elif mode == 'mad':
                # Mean absolute deviation for this group
                var_mean = var_mean + tf.math.reduce_mean(tf.math.abs(curr_losses - tf.math.reduce_mean(curr_losses)))

    # Apply the weight and give the operation a name
    return tf.math.multiply(var_mean, weight, name='dissipation_variation_loss')
