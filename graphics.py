"""
A module for graphics / visualizations / plots / animations etc.
"""
import math
import os
import logging

import numpy as np
from matplotlib import pyplot as plt, animation as animation
from sklearn.decomposition import PCA, IncrementalPCA

from analysis import propagate_x_and_z, get_compound_model_components
from utils import save_model, log_and_time
from therm import diffusion_entropy_rate_vs_time, gray_scott_reaction_entropy_rate_vs_time, \
    reaction_network_entropy_rate_vs_time


def get_x_display_range(x):
    """
    Return the color limits for displaying the input tensor (X domain)
    Args:
        x: tensor to get display range.

    Returns:
        (min, max) tuple for display range.

    """

    return (np.percentile(x, 0.1), np.percentile(x, 99.9))


def show_x_of_t(x_vs_t, t, i=0, axes=plt, **kwargs):
    """
    Shows a single X domain at a single timepoint, after rescaling values for visualization.
    Args:
        x_vs_t: (Nt, batch, height, width, channels) tensor of Xs
        t: the timepoint to show
        i: which sample from the batch to show.
        axes: axes handle to plot on
        **kwargs: args to pass to scale_im

    Returns:
        im_han: handle to image.

    """

    im_han = axes.imshow(prep_im(x_vs_t[t, i, :, :, :], **kwargs))

    return im_han


def show_x_of_t_batch(x_vs_t, t, n_show=9, **kwargs):
    """
    Shows some subset of a batch of X at a particular timepoint as a gallery of sub-plots.
    Args:
        x_vs_t: (Nt, batch, height, width, channels) tensor of Xs
        t: timepoint to show
        n_show: show up to n_show samples in a montage
        **kwargs: args to pass down the call stack.

    Returns:
        fig_han: handle to figure.

    """
    n_show = min(n_show, x_vs_t.shape[1])

    w = int(math.ceil(math.sqrt(n_show)))
    fig_han, axes = plt.subplots(w, w, figsize=(4*w, 4*w))
    axes = np.array(axes)  # Handle the single-image case

    for i_show in range(n_show):
        show_x_of_t(x_vs_t, t, i=i_show, axes=axes.flatten()[i_show],
                    x_range=get_x_display_range(x_vs_t[t, i_show]), **kwargs)
    return fig_han


def save_x_of_t_batch_snapshots(x_vs_t, ts, out_dir, **kwargs):
    """
    Calls show_x_of_t_batch on several timepoints and saves the resulting figures.

    Args:
        x_vs_t: (Nt, batch, height, width, channels) tensor of Xs
        ts: list of timepoints to show
        **kwargs: args to pass down the call stack.

    Returns:
        None
    """
    for t in ts:
        fig = show_x_of_t_batch(x_vs_t, t, **kwargs)
        fig.savefig(os.path.join(out_dir, 'X_' + str(t) + '.png'))


def animate_x_vs_t(x_vs_t, i=0, contrast=1.0, x_range=None):
    """
    Animates a single X over time.
    Args:
        x_vs_t: (Nt, batch, height, width, channels) tensor of Xs
        i: index of sample from batch to animate.
        contrast: to pass to scale_im
        x_range: the expected range of X, if it should remain fixed over time.

    Returns:
        anim: handle to animation object.

    """
    # Animates a single X state over time.
    fig = plt.figure()
    n_t = x_vs_t.shape[0]
    im_han = show_x_of_t(x_vs_t, 0, i=i)
    anim = animation.FuncAnimation(fig, _update_x_anim, fargs=(contrast, im_han, x_vs_t, i, x_range),
                                   frames=n_t, interval=125)
    return anim


def animate_x_vs_t_batch(x_vs_t, out_dir, n_show=2, save=True, dpi=100, **kwargs):
    """
    Animates a subset of a batch of Xs over time and saves to disk.
    Args:
        x_vs_t: (Nt, batch, height, width, channels) tensor of Xs
        out_dir: Directory to save animations to as .mov
        n_show: Animate up to n_show samples from X
        save: If true, save to disk.
        dpi: resolution to save at, will be passed to matplotlib animation.save
        **kwargs: args to pass down the call stack.

    Returns:
        None
    """
    n_show = min(n_show, x_vs_t.shape[1])

    # Animate a few samples of X over time.
    for i_show in range(n_show):
        anim = animate_x_vs_t(x_vs_t, i=i_show, **kwargs)
        if save:
            anim_file = os.path.join(out_dir, 'animation_X' + str(i_show) + '.mp4')
            save_x_anim(anim, anim_file, dpi=dpi)


def _update_x_anim(t, contrast, im_han, x_vs_t, i_show, x_range):

    if not x_range:
        x_range = get_x_display_range(x_vs_t[t, i_show])
    im_han.set_array(prep_im(x_vs_t[t, i_show, :, :, :], contrast=contrast, x_range=x_range))


def save_x_anim(anim, out_file_path, dpi=100):
    """
    Saves an animation to disk.
    Args:
        anim: animation object.
        out_file_path: path to save to.
        dpi: resolution to pass to animation.save

    Returns:
        None
    """

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, metadata=dict(artist='HunterElliott'), bitrate=-1)
    anim.save(out_file_path, writer=writer, dpi=dpi)


def prep_im(x, **kwargs):
    """
    Applies some standard pre-processing so that the input X is suitable for display as an RGB image.
     This includes padding if the number of channels is <3 and projection via PCA if > 3
    Args:
        x: (H, W, C) state space to prep for display as image
        **kwargs: kwargs to pass on to e.g. scale_im

    Returns:
        x: (H, W, 3) image in range (0,1), possibly contrast-enhanced, etc.

    """
    if x.shape[-1] == 2:
        # For two channels we still use RGB with one blank channel
        x = np.concatenate([np.zeros(shape=x.shape[0:2] + (3-x.shape[-1],)), x], axis=-1)
    elif x.shape[-1] > 3:
        # For > 3 channels we project down to 3.
        pca = PCA(n_components=3, svd_solver='randomized')
        x_shape = x.shape
        x = np.reshape(pca.fit_transform(np.reshape(x, (-1, x_shape[-1]))), x_shape[0:-1] + (3,))

    x = scale_im(x, **kwargs)

    return x


def scale_im(x, contrast=-10.0, x_range=(-1, 1)):
    """
    Maps the input image to (0,1) range and scales for visualization, clipping values outside (0, 1)
    Args:
        x: the image to scale
        contrast: If > 0, linearly scales by this factor, if <0 a sigmoidal scaling is applied to improve dynamic range.
        x_range: The range within which the input X was bounded.

    Returns:
        x: x re-scaled and clipped to (0, 1)

    """

    # We always map to (0, 1) for visualization
    x = x - x_range[0]
    x = x / (x_range[1] - x_range[0])

    if contrast > 0:
        x = x * contrast
    else:
        x = 1 / (1 + np.exp(-(x * -contrast - 4)))

    # Avoid complaints from plotting functions by clipping values.
    x = np.clip(x, 0, 1)
    return x


def project_channels_to_rgb(x):
    """
    Projects the channel (last) dimension of the input array down to 3 dimensions via PCA so that it can be displayed
    as an RGB image.
    Args:
        x: (..., n_channels) array with n_channels>3

    Returns:
        x: (..., 3) projected version of x

    """

    t_start = log_and_time("Projecting down to 3 channels via PCA for visualization...")

    # Project using online PCA to keep the memory footprint bounded
    x_shape = x.shape
    projector = IncrementalPCA(n_components=3, batch_size=256)
    x = np.reshape(projector.fit_transform(np.reshape(x, (-1, x_shape[-1]))), x_shape[0:-1] + (3,))

    log_and_time(t_start)
    logging.info("Projected components explain {} netting {} of the variance."
                 .format(projector.explained_variance_ratio_, np.sum(projector.explained_variance_ratio_)))

    return x


def show_z_dist_vs_t_batch(z_dist_vs_t):
    """
    Plots a superposition of all the input Z-distance-vs-time curves
    Args:
        z_dist_vs_t: (Nt, batch size) matrix of Z-distance-vs-time curves.

    Returns:
        fig_han: handle to figure.

    """

    fig = plt.figure()
    for i in range(z_dist_vs_t.shape[1]):
        plt.plot(z_dist_vs_t[:, i], alpha=0.5)

    mean_z_dist = np.mean(z_dist_vs_t, axis=1)
    plt.plot(mean_z_dist, color='black', linewidth=3)

    plt.xlabel('Iterations')
    plt.ylabel('D(Z, Z0)')

    return fig

# ---- Composite / model-specific functions ---- #


def save_propagation_model_and_figures(model, out_dir, z_bits=None):

    # Save the model
    save_model(model, out_dir)

    # Run a batch through and output at every t to make figures
    x_vs_t, z_vs_t, z_acc_vs_t, z_dist_vs_t = propagate_x_and_z(model, z_bits=z_bits)
    if any(['encoder' in output_name for output_name in model.output_names]) and x_vs_t.shape[0] > 1:
        save_z_propagation_figures(z_dist_vs_t, z_acc_vs_t, out_dir)

    save_therm_figures(model, x_vs_t, out_dir)
    save_propagation_figures(x_vs_t, out_dir)


def plot_dissipation_rate(mean_diss, std_diss, d_t=1, fig=None, logy=False):

    t = np.arange(0, mean_diss.shape[0] * d_t, d_t)
    if fig is None:
        fig = plt.figure()
    plt.plot(t, mean_diss)
    plt.fill_between(t, mean_diss - std_diss, mean_diss + std_diss, alpha=.5)
    plt.xlabel("Time, seconds")
    plt.ylabel("Entropy Production Rate (J*M)/(K*s)")

    return fig


def plot_correlation_vs_time(mean_corr, std_corr, d_t=1, fig=None, logy=False):

    t = np.arange(0, mean_corr.shape[0] * d_t, d_t)
    if fig is None:
        fig = plt.figure()
    plt.plot(t, mean_corr)
    plt.fill_between(t, mean_corr - std_corr, mean_corr + std_corr, alpha=.5)
    plt.xlabel("Time, seconds")
    plt.ylabel(r"Correlation, $\rho$")

    return fig


def save_therm_figures(model, x_vs_t, out_dir):

    transition_model = get_compound_model_components(model)[1]
    x_vs_t = transition_model.de_center_x(x_vs_t)

    transition_model.log_free_parameters()

    mean_diff_ds, std_diff_ds = diffusion_entropy_rate_vs_time(x_vs_t, transition_model.diff_coeffs)
    fig = plot_dissipation_rate(mean_diff_ds, std_diff_ds, d_t=transition_model.d_t)
    fig.savefig(os.path.join(out_dir, 'diffusion_ds_vs_t.png'))

    if 'gray_scott' in transition_model.name:
        # TODO - unify this with CRN / reversible methods...
        assert np.all(transition_model.get_decay_rate(0) == transition_model.get_decay_rate(1)), \
            "Temporally variable decay rates not yet supported!"
        mean_rxn_ds, std_rxn_ds = gray_scott_reaction_entropy_rate_vs_time(x_vs_t, transition_model.decay_rate)
    else:
        mean_rxn_ds, std_rxn_ds = reaction_network_entropy_rate_vs_time(x_vs_t,
                                                                        transition_model.reactants,
                                                                        transition_model.products,
                                                                        transition_model.rate_const)

    fig = plot_dissipation_rate(mean_rxn_ds, std_rxn_ds, d_t=transition_model.d_t)
    fig.savefig(os.path.join(out_dir, 'reaction_ds_vs_t.png'))


def save_propagation_figures(x_vs_t, out_dir, **kwargs):

    # First save separate figures for each channel
    for i_chan in range(x_vs_t.shape[-1]):
        chan_out_dir = os.path.join(out_dir, 'Channel_' + str(i_chan))
        os.makedirs(chan_out_dir)
        save_x_vs_t_figure_sets(np.expand_dims(x_vs_t[:, :, :, :, i_chan], -1), chan_out_dir, **kwargs)

    # Save a merged version as well
    if x_vs_t.shape[-1] > 3:
        # Pre-project the whole timeseries to ensure temporal consistency
        x_vs_t = project_channels_to_rgb(x_vs_t)

    save_x_vs_t_figure_sets(x_vs_t, out_dir, **kwargs)


def save_x_vs_t_figure_sets(x_vs_t, out_dir, **kwargs):

    # Plot a few samples from the batch at t=0 and t=T
    save_x_of_t_batch_snapshots(x_vs_t, [0, x_vs_t.shape[0] - 1], out_dir, **kwargs)

    if x_vs_t.shape[0] > 1:
        # If a time series, animate it and save plots.
        animate_x_vs_t_batch(x_vs_t, out_dir, **kwargs)


def save_z_propagation_figures(z_dist_vs_t, z_acc_vs_t, out_dir):

    fig = show_z_dist_vs_t_batch(z_dist_vs_t)
    fig.savefig(os.path.join(out_dir, 'Z_dist_vs_t.png'))

    fig = plt.figure()
    plt.plot(z_acc_vs_t, linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Mean Z Accuracy')
    plt.ylim(0, 1)
    fig.savefig(os.path.join(out_dir, 'Z_accuracy_vs_t.png'))


def save_replication_gallery(x_output, out_dir, file_name='X012_montage.png', **kwargs):
    x_montage = np.concatenate([x_output[:, :, :, :, 0],
                                x_output[:, :, :, :, 1],
                                x_output[:, :, :, :, 2]], axis=1)

    fig = show_x_of_t_batch(np.expand_dims(x_montage, 0), 0, **kwargs)
    fig.savefig(os.path.join(out_dir, file_name))
