"""
Control experiments to confirm the necessity/utility for various components resulting from the optimizations.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import analysis
import graphics
import utils
import models
import therm
import sampling



# model_dirs = ['/Users/hunterelliott/Iteration_C1/Dense_CRNs/v2_Max_Diss/test_27/Nt64',
#               '/Users/hunterelliott/Iteration_C1/Dense_CRNs/v2_Max_Diss/test_29/Nt64',
#               '/Users/hunterelliott/Iteration_C1/Dense_CRNs/v1_Fixed_Ilya/test_173/Nt32',
#               '/Users/hunterelliott/Iteration_C1/Dense_CRNs/v1_Fixed_Ilya/test_177/Nt64']

model_dirs = ['/media/hunter/fast storage/Training_Experiments/Iteration_C1/Dense_CRNs/v2_Max_Diss/test_27/Nt64',
              '/media/hunter/fast storage/Training_Experiments/Iteration_C1/Dense_CRNs/v2_Max_Diss/test_29/Nt64',
              '/media/hunter/fast storage/Training_Experiments/Iteration_C1/Dense_CRNs/v1_Fixed_Ilya/test_173/Nt32',
              '/media/hunter/fast storage/Training_Experiments/Iteration_C1/Dense_CRNs/v1_Fixed_Ilya/test_177/Nt64']

target_path = '../assets/Ilya_Prigogine_1977c_compressed.jpg'

analysis_funcs = [lambda x_vs_t, model : therm.diffusion_entropy_rate_vs_time(x_vs_t, model.diff_coeffs),
                  lambda x_vs_t, model : therm.diffusion_entropy_rate_vs_time(x_vs_t, model.diff_coeffs),
                  lambda x_vs_t, model : analysis.pearsons_corr_vs_time(x_vs_t, utils.load_and_normalize_image(x_shape, target_path),  (0, 1)),
                  lambda x_vs_t, model : analysis.pearsons_corr_vs_time(x_vs_t, utils.load_and_normalize_image(x_shape, target_path),  (0, 1))]
plot_funcs = [graphics.plot_dissipation_rate, graphics.plot_correlation_vs_time, graphics.plot_correlation_vs_time, graphics.plot_correlation_vs_time]

clip_y_lim = [(0.0, 3e-23), (0.0, 3e-23), (-.25, 1.0), (-.25, 1.0)]
min_y_lim = [0.0, 0.0, -1.0, -1.0]
max_y_lim = [5e-22, 5e-22, 1.0, 1.0]
log_clip_y_lim = [(2e-25, 2e-21), (2e-25, 2e-21), (.75, 1.0), (.75, 1.0)]
override_T = [100, 100, 100, 100]

metric_names = ['dissipation', 'dissipation', 'correlation', 'correlation']

n_metrics = len(metric_names)

z_bits = 64
batch_size = 32
d_t = .01  # To avoid numerical error in random kinetics models we optionally adjust d_t

for i_metric in range(n_metrics):

    model = utils.load_model(model_dirs[i_metric])
    generator_model, transition_model, iteration_layer, encoder_model = analysis.get_compound_model_components(model)
    x_shape = generator_model.output_shape[1:]
    n_t_og = iteration_layer.n_t  # This is in seconds
    if override_T[i_metric] is not None:
        n_t = round(override_T[i_metric] / d_t)
    else:
        n_t = n_t_og

    # out_dir = os.path.join(model_dirs[i_metric], 'controlsv2' + '_test' + str(random.randint(1000, 9999)))
    out_dir = os.path.join(model_dirs[i_metric], 'controlsv2' + '_dt' + str(d_t) + '_nt' + str(n_t) + '_batch' + str(batch_size))
    utils.prep_output_dir(out_dir)
    utils.posterity_log(out_dir, locals(), __file__)

    # ---- Baseline - Optimized Model ---- #
    # Get metric for actual optimized model
    transition_model.d_t = d_t # Optionally adjust d_t
    x_0_og = generator_model(sampling.get_train_z(generator_model.input_shape[1:], batch_size, entropy=z_bits))
    x_vs_t = transition_model.de_center_x(analysis.propagate_x(x_0_og, transition_model, n_t))
    mean_diff_ds_og, std_diff_ds_og = analysis_funcs[i_metric](x_vs_t, transition_model)

    anim = graphics.animate_x_vs_t(x_vs_t)
    graphics.save_x_anim(anim, os.path.join(out_dir, 'result_animation.mp4'))

    # ---- Structure Control ---- #
    # Compare metric with same kinetics but different X_0 structures

    # Match first moment statistics to original X0 so structure is primary difference.
    x_0_og_mean = np.mean(x_0_og)
    x_0_og_std = np.std(x_0_og)
    x_0 = np.random.normal(x_0_og_mean, x_0_og_std, size=x_0_og.shape)

    x_vs_t = transition_model.de_center_x(analysis.propagate_x(x_0, transition_model, n_t))
    mean_diff_ds_struc, std_diff_ds_struc = analysis_funcs[i_metric](x_vs_t, transition_model)

    anim = graphics.animate_x_vs_t(x_vs_t)
    graphics.save_x_anim(anim, os.path.join(out_dir, 'structure_control_animation.mp4'))

    # ---- Initialization Control ---- #
    # Compare metric with same X_0 but the initial kinetics

    # Get a batch where each time series is from a randomly initialized transition model
    x_vs_t = []
    diff_ds = []
    for i_samp in range(batch_size):
        logging.info("Running sample {} with initialized transition model.".format(i_samp))
        transition_model_rnd = models.transition_model('dense_gs_crn', n_species=x_0_og.shape[-1])
        transition_model_rnd.d_t = d_t # Use the global d_t
        x_vs_t.append(transition_model.de_center_x(analysis.propagate_x(np.expand_dims(x_0_og[1], axis=0),
                                                                        transition_model_rnd, n_t)))
        tmp, _ = analysis_funcs[i_metric](x_vs_t[i_samp], transition_model_rnd)
        diff_ds.append(np.expand_dims(tmp, axis=0))

    # Combine and get moments
    x_vs_t = np.concatenate(x_vs_t, axis=1)
    diff_ds = np.concatenate(diff_ds, axis=0)
    mean_diff_ds_ini = np.mean(diff_ds, axis=0)
    std_diff_ds_ini = np.std(diff_ds, axis=0)


    anim = graphics.animate_x_vs_t(x_vs_t)
    graphics.save_x_anim(anim, os.path.join(out_dir, 'initialization_control_animation.mp4'))

    # ---- Kinetics Control ---- #
    # Compare metric with same X_0 but random kinetics

    # Get a batch where each time series is from a random kinetics transition model
    x_vs_t = []
    diff_ds = []
    for i_samp in range(batch_size):
        logging.info("Running sample {} with randomized transition model.".format(i_samp))

        # It's not obvious that there's a truly "fair" way to do this so we sample again matching moments but ensuring
        # validity. Many of these may be highly unstable...
        rnd_rate_const = np.random.normal(np.mean(transition_model.rate_const), np.std(transition_model.rate_const), size=transition_model.rate_const.shape)
        rnd_rate_const[rnd_rate_const < 0] = 1e-10  # Ensure validity of rate and resulting diss rate
        rnd_diff_coef = np.random.normal(np.mean(transition_model.diff_coeffs), np.std(transition_model.diff_coeffs), size=transition_model.diff_coeffs.shape)
        # We only have numerical stability within this range
        rnd_diff_coef[rnd_diff_coef < .05] = .05
        rnd_diff_coef[rnd_diff_coef > .2] = .2

        rnd_flow_rate = np.random.uniform(0, .08) # We have no statistics for this so we use pearson's "interesting" range.

        rnd_feed_conc = np.random.normal(np.mean(transition_model.drive.feed_conc),
                                         np.std(transition_model.drive.feed_conc),
                                         size=transition_model.drive.feed_conc.shape)
        rnd_feed_conc[rnd_feed_conc < 0] = 0
        drive_kwargs = {'init_feed_conc': rnd_feed_conc, 'init_flow_rate': rnd_flow_rate}
        drive_rnd = models.FlowDriveLayer(**drive_kwargs)
        transition_model_rnd = models.ReactionNetworkTransitionLayer(reactants=transition_model.reactants,
                                                                     products=transition_model.products,
                                                                     init_rate_const=rnd_rate_const,
                                                                     init_diff_coeffs=rnd_diff_coef,
                                                                     drive=drive_rnd)

        transition_model_rnd.d_t = d_t

        x_vs_t.append(transition_model.de_center_x(analysis.propagate_x(np.expand_dims(x_0_og[1], axis=0),
                                                                        transition_model_rnd, n_t)))
        tmp, _ = analysis_funcs[i_metric](x_vs_t[i_samp], transition_model_rnd)
        diff_ds.append(np.expand_dims(tmp, axis=0))

    # Combine and get diss stats
    x_vs_t = np.concatenate(x_vs_t, axis=1)
    diff_ds = np.concatenate(diff_ds, axis=0)
    mean_diff_ds_kin = np.mean(diff_ds, axis=0)
    std_diff_ds_kin = np.std(diff_ds, axis=0)


    anim = graphics.animate_x_vs_t(x_vs_t)
    graphics.save_x_anim(anim, os.path.join(out_dir, 'kinetics_control_animation.mp4'))


    # ---- Overlay Figure ---- #

    fig = plot_funcs[i_metric](mean_diff_ds_og, std_diff_ds_og, d_t=transition_model.d_t)
    fig = plot_funcs[i_metric](mean_diff_ds_struc, std_diff_ds_struc, d_t=transition_model.d_t, fig=fig)
    fig = plot_funcs[i_metric](mean_diff_ds_ini, std_diff_ds_ini, d_t=transition_model.d_t, fig=fig)
    fig = plot_funcs[i_metric](mean_diff_ds_kin, std_diff_ds_kin, d_t=transition_model.d_t, fig=fig)
    fig.gca().legend(['Optimized', 'Random $X_0$', r'Initial $\theta_C$', r'Random $\theta_C$'])
    og_ylim = fig.gca().get_ylim()

    #plt.show()
    fig.gca().set_xlim((0, n_t*d_t))
    fig.gca().set_ylim((max(min_y_lim[i_metric], og_ylim[0]), min(og_ylim[1], max_y_lim[i_metric])))
    fig.savefig(os.path.join(out_dir, metric_names[i_metric] + '_traj_overlay.png'), dpi=300)
    fig.gca().set_ylim(clip_y_lim[i_metric])
    fig.savefig(os.path.join(out_dir, metric_names[i_metric] + '_traj_overlay_clipy.png'), dpi=300)
    fig.gca().set_yscale('log')
    fig.gca().set_ylim(og_ylim)

    fig.savefig(os.path.join(out_dir, metric_names[i_metric] + '_traj_overlay_logy.png'), dpi=300)
    fig.gca().set_ylim(log_clip_y_lim[i_metric])
    fig.savefig(os.path.join(out_dir, metric_names[i_metric] + '_traj_overlay_logy_clipy.png'), dpi=300)

    # Save the raw data as well
    np.save(os.path.join(out_dir, metric_names[i_metric] + '_optimized.npy'), mean_diff_ds_og)
    np.save(os.path.join(out_dir, metric_names[i_metric] + '_randomx0.npy'), mean_diff_ds_struc)
    np.save(os.path.join(out_dir, metric_names[i_metric] + '_randinit.npy'), mean_diff_ds_ini)
    np.save(os.path.join(out_dir, metric_names[i_metric] + '_randkinetics.npy'), mean_diff_ds_kin)

    np.save(os.path.join(out_dir, metric_names[i_metric] + '_optimized_std.npy'), std_diff_ds_og)
    np.save(os.path.join(out_dir, metric_names[i_metric] + '_randomx0_std.npy'), std_diff_ds_struc)
    np.save(os.path.join(out_dir, metric_names[i_metric] + '_randinit_std.npy'), std_diff_ds_ini)
    np.save(os.path.join(out_dir, metric_names[i_metric] + '_randkinetics_std.npy'), std_diff_ds_kin)

logging.info("Done!")






