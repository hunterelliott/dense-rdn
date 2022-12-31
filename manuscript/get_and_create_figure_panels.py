"""
Render/copy figure panels to central location for making figures (in code for posterity).
"""

import os
import shutil
import logging

import matplotlib.pyplot as plt
import numpy as np

import utils
import graphics
import analysis
import models
import domains

dest_dir = '/Users/hunterelliott/Documents/Iteration_Manuscript_Figures_Temp'
utils.configure_logging(dest_dir)


def get_dest_file_name(dest_dir, fig_name, panel_name, file_ext='.png'):
    return os.path.join(dest_dir, 'Figure_' + fig_name + '_Panel_' + panel_name + file_ext)


def copy_panel(source_file, dest_dir, fig_name, panel_name):
    _, f_ext = os.path.splitext(source_file)
    dest_file = get_dest_file_name(dest_dir, fig_name, panel_name, f_ext)
    logging.info("Copying {} to {}".format(source_file, dest_file))
    shutil.copy2(source_file, dest_file)


def save_panel(figure, dest_dir, fig_name, panel_name, dpi=None):
    dest_file = get_dest_file_name(dest_dir, fig_name, panel_name, '.png')
    logging.info("Saving panel to {}".format(dest_file))
    figure.savefig(dest_file, bbox_inches='tight', dpi=dpi)


def save_time_series(x_vs_t, i_ts, fig_name, panel_name, im_range, i_chan, i_show=0):

    i_chan = np.array(i_chan)
    for i_t in i_ts:
        im = x_vs_t[i_t, i_show][:, :, i_chan]
        # im = x_vs_t[i_t, i_show]
        # im = im[:,:, i_chan]
        # if len(i_chan) == 1:
        #     im = np.expand_dims(im[:, :, 0], -1)
        fig = plt.figure()
        plt.imshow(graphics.prep_im(im, x_range=im_range))
        plt.axis('off')
        save_panel(fig, dest_dir, fig_name, panel_name + '_T' + str(i_t))


make_figs = ['FixedIlya']

## ----===== Complex Dynamics Example Fig ====---- ##

fig_name = 'CompExamp'

if fig_name in make_figs:

    decay_rates = (.056, .059)
    coupling_rate = 1e-3
    crn = models.coupled_gray_scott_crn(decay_rates=decay_rates, coupling_rate=coupling_rate)
    w = 128
    g_w = int(w / 3)
    h_w = int(w / 8)
    x_shape = (w, w, crn.n_species)
    x_0 = np.zeros((2,) + x_shape, dtype=np.float32)
    # We mimic Pearson's initial conditions with a second seed region for the second autocatalyst/substrate pair
    # Substrate everywhere in background
    x_0[:, :, :, 0] = 1.0
    x_0[:, :, :, 2] = 1.0
    # Substrate-autocatalyst A mixture in big square
    x_0[:, int(w / 2) - g_w:int(w / 2) + g_w, int(w / 2) - g_w:int(w / 2) + g_w, 0] = 0.5
    x_0[:, int(w / 2) - g_w:int(w / 2) + g_w, int(w / 2) - g_w:int(w / 2) + g_w, 1] = 0.25
    # Substrate-autocatalyst B mixture in small off-center rectangle
    x_0[:, int(w / 2) - h_w:int(w / 2) + h_w, int(w / 2) - h_w:int(w / 2) + g_w, 2] = 0.5
    x_0[:, int(w / 2) - h_w:int(w / 2) + h_w, int(w / 2) - h_w:int(w / 2) + g_w, 3] = 0.25
    # Noise to break symmetry
    x_0 = x_0 * np.random.uniform(.95, 1.05, x_0.shape)
    x_0 = crn.center_x(x_0)

    #n_t = 2500
    n_t = 1000
    n_snap = 9
    i_ts = [int(n_t / (n_snap-1) * i) for i in range(n_snap)]
    x_vs_t = analysis.propagate_x(x_0, crn, n_t)

    x_vs_t = graphics.project_channels_to_rgb(x_vs_t)
    im_range = graphics.get_x_display_range(x_vs_t)

    save_time_series(x_vs_t, i_ts, fig_name, 'ResultATargetChan', im_range, np.arange(0, x_vs_t.shape[-1]))


## ----===== Fixed Ilya Figure ====---- ##

fig_name = 'FixedIlya'

if fig_name in make_figs:

    ## -- Target Panel -- ##

    w = 64
    fig = plt.figure()
    im = utils.load_and_normalize_image((w, w, 1), '../assets/Ilya_Prigogine_1977c_compressed.jpg')
    im_range = graphics.get_x_display_range(im)
    plt.imshow(graphics.prep_im(np.expand_dims(im[:, :, 0], -1), x_range=im_range))
    plt.axis('off')
    save_panel(fig, dest_dir, fig_name, 'Target')

    ## -- Result Panels -- ##

    i_ts = [0, 100, 1000, 10000]
    #i_ts = [0, 100]

    model_dir = '/Users/hunterelliott/Iteration_C1/Dense_CRNs/v1_Fixed_Ilya/test_173/Nt32'
    model = utils.load_model(model_dir)
    n_t = max(i_ts) + 1
    d_t = 1
    z_bits = 64
    x_vs_t, _, _, _ = analysis.propagate_x_and_z(model, n_t=n_t, d_t=d_t, z_bits=z_bits, batch_size=1)

    # Correlation vs time
    # We're compositing these in keynote now because multi-panel in latex is annoying, but recording paths here for
    # posterity:
    # Example A:
    fig_path_example_a = "/Users/hunterelliott/Iteration_C1/Dense_CRNs/v1_Fixed_Ilya/test_173/Nt32/controlsv2_dt0.1_nt10000_batch32"
    # Example B:
    fig_path_example_b = "/Users/hunterelliott/Iteration_C1/Dense_CRNs/v1_Fixed_Ilya/test_177/Nt64/controlsv2_dt0.1_nt10000_batch32"


    # Per-channel image time series
    save_time_series(x_vs_t, i_ts , fig_name, 'ResultATargetChan', im_range, (1,))

    # Projected image time series
    x_vs_t = graphics.project_channels_to_rgb(x_vs_t)
    im_range = graphics.get_x_display_range(x_vs_t)

    save_time_series(x_vs_t, i_ts, fig_name, 'ProjResultA', im_range, np.arange(0, x_vs_t.shape[-1]),
                     i_show=0)


    model_dir = '/Users/hunterelliott/Iteration_C1/Dense_CRNs/v1_Fixed_Ilya/test_177/Nt64'
    model = utils.load_model(model_dir)
    d_t = 1
    z_bits = 64
    x_vs_t, _, _, _ = analysis.propagate_x_and_z(model, n_t=n_t, d_t=d_t, z_bits=z_bits, batch_size=1)

    save_time_series(x_vs_t, i_ts, fig_name, 'ResultBTargetChan', im_range, (1,))

    x_vs_t = graphics.project_channels_to_rgb(x_vs_t)
    im_range = graphics.get_x_display_range(x_vs_t)

    save_time_series(x_vs_t, i_ts, fig_name, 'ProjResultB', im_range, np.arange(0, x_vs_t.shape[-1]),
                     i_show=0)


## ----===== Dissipation Maximization Figure ====---- ##

fig_name = 'DissMax'


if fig_name in make_figs:


    # -- Example Reaction Graph -- ##
    # For the appendix
    panel_name = 'ReactionGraph'
    copy_panel('/Users/hunterelliott/Iteration_C1/Dense_CRNs/v2_Max_Diss/test_1/Nt32/reaction_graph.png',
               dest_dir, fig_name, panel_name)


    ## -- Example t0 panels -- ##

    # Load and get X0 from both so we can do a common projection

    model_dir = '/Users/hunterelliott/Iteration_C1/Dense_CRNs/v2_Max_Diss/test_27/Nt64'
    model = utils.load_model(model_dir)
    x_vs_t, _, _, _ = analysis.propagate_x_and_z(model, n_t=8, d_t=1, z_bits=64, batch_size=1)
    model_dir = '/Users/hunterelliott/Iteration_C1/Dense_CRNs/v2_Max_Diss/test_29/Nt64'
    model = utils.load_model(model_dir)
    x_vs_t_2, _, _, _ = analysis.propagate_x_and_z(model, n_t=8, d_t=1, z_bits=64, batch_size=1)
    x_vs_t = np.concatenate([x_vs_t, x_vs_t_2], axis=1)
    x_vs_t = graphics.project_channels_to_rgb(x_vs_t)
    im_range = graphics.get_x_display_range(x_vs_t)

    fig = plt.figure()
    plt.imshow(graphics.prep_im(x_vs_t[0, 0, :, :, :], x_range=im_range))
    plt.axis('off')
    save_panel(fig, dest_dir, fig_name, 'ProjResultA')

    fig = plt.figure()
    plt.imshow(graphics.prep_im(x_vs_t[0, 1, :, :, :], x_range=im_range))
    plt.axis('off')
    save_panel(fig, dest_dir, fig_name, 'ProjResultB')


    # -- Dissipation Rate Comparison Panels -- ##
    panel_name = 'DissCompare'
    copy_panel('/Users/hunterelliott/Iteration_C1/Dense_CRNs/v2_Max_Diss/test_27/Nt64/controls_dt0.01_nt6400/diff_diss_rate_traj_overlay_clipy.png',
               dest_dir, fig_name, panel_name)


## ----===== Information Transmitting Dissipative Struct Figure ====---- ##

fig_name = 'DissInf'

if fig_name in make_figs:
    ## -- Example time-series -- ##

    model_dir = '/Users/hunterelliott/Iteration_C1/Dense_CRNs/v3_DissInf/test_53/Nt128'
    model = utils.load_model(model_dir)
    x_vs_t, _, _, _ = analysis.propagate_x_and_z(model, n_t=1024, d_t=1, z_bits=64, batch_size=3)
    x_vs_t = graphics.project_channels_to_rgb(x_vs_t)
    im_range = graphics.get_x_display_range(x_vs_t)

    save_time_series(x_vs_t, [10, 100, 1000], fig_name, 'ProjResultA', im_range, np.arange(0, x_vs_t.shape[-1]), i_show=0)
    save_time_series(x_vs_t, [10, 100, 1000], fig_name, 'ProjResultB', im_range, np.arange(0, x_vs_t.shape[-1]), i_show=1)
    save_time_series(x_vs_t, [10, 100, 1000], fig_name, 'ProjResultC', im_range, np.arange(0, x_vs_t.shape[-1]), i_show=2)

## ----===== Replication Figure ====---- ##

fig_name = 'Rep'

if fig_name in make_figs:

    ## -- Domain structure illustration -- ##

    x_shape = (32, 32, 3)
    x_buffer = 8
    yard = 0
    rd = domains.ReplicationDomain(x_shape, buffer=x_buffer, yard=yard)

    fig = plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim((0, rd.full_domain.shape[1]))
    ax.set_ylim((0, rd.full_domain.shape[0]))

    domains.draw_domain(rd.full_domain)

    domains.draw_domain(rd.parent_domain)
    domains.draw_domain(rd.daughter_domains[0])
    domains.draw_domain(rd.daughter_domains[1])
    save_panel(fig, dest_dir, fig_name, 'DomainDiag', dpi=200)



    ## -- Example time-series -- ##

    model_dir = '/Users/hunterelliott/Iteration_C1/Dense_CRNs/v4_Replication/test_84'
    model = utils.load_model(model_dir)
    x_vs_t, _, _, _ = analysis.propagate_x_and_z(model, n_t=352, d_t=1, z_bits=24, batch_size=1)
    x_vs_t = graphics.project_channels_to_rgb(x_vs_t)
    im_range = graphics.get_x_display_range(x_vs_t)

    save_time_series(x_vs_t, [round(352/3*x) for x in range(4)], fig_name, 'ProjResultA', im_range, np.arange(0, x_vs_t.shape[-1]), i_show=0)

    # -- Example Reaction Graph -- ##
    # For the appendix
    panel_name = 'ReactionGraph'
    copy_panel('/Users/hunterelliott/Iteration_C1/Dense_CRNs/v4_Replication/test_84/reaction_graph.png',
               dest_dir, fig_name, panel_name)

