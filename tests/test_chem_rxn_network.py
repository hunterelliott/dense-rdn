"""
Tests specific to the chemical reaction network transition layer
"""
import math

import numpy as np
import pytest

import chem
import models
from models.transition import ReactionNetworkTransitionLayer
from models.drive import FlowDriveLayer, ConstantSynthesisDriveLayer
from analysis import propagate_x


@pytest.fixture(params=[.062, .03])
def gs_k(request):
    return request.param


@pytest.fixture(params=[models.transition.reversible_gray_scott_crn(),
                        models.transition.reversible_coupled_gray_scott_crn(),
                        models.transition.dense_crn(4, 'create_dense_reversible_unimolecular_reaction_network_stoichiometry'),
                        models.transition.dense_crn(3, 'create_dense_reversible_autocatalytic_reaction_network_stoichiometry'),
                        models.transition.dense_crn(4, 'create_dense_bimolecular_reaction_network_stoichiometry'),
                        models.transition.dense_crn(5, 'create_dense_2_species_third_order_reaction_network_stoichiometry'),
                        models.transition.dense_crn(4, 'create_dense_2_species_mixed_order_reaction_network_stoichiometry'),
                        models.transition.dense_crn(5, 'create_dense_mixed_autocatalytic_reaction_network_stoichiometry'),
                        models.transition.dense_gs_crn(4),
                        models.transition.dense_kitchen_sink_crn(4),
                        ])
def reversible_crn_instance(request):
    return request.param


def numpy_x_0_for_crn(crn):
    return np.random.uniform(-.1, .1, size=(3, 32, 32, crn.n_species)).astype(np.float32)


def test_crn_vs_gray_scott(tiny_numpy_x_0, gs_k):
    """
    Configure the CRN to be equivalent to the gray scott model and then compare
    """

    F = .033
    k = gs_k
    d = (.2, .1)
    d_t = 1.0

    # Run a good number of iterations for strict comparison but on a small region for speed
    n_t = 128

    reactants = np.array([[1, 0],
                          [2, 1]])

    products = np.array([[0, 0],
                         [3, 0]])

    # The non-dimensionalized gray scott assumes k=1 for autocatalytic reaction
    k_rxn = [1.0, k]
    # It also assumes only U is fed and at a concentration of 1
    feed_conc = (1.0, 0.0)
    drive_layer = FlowDriveLayer(init_feed_conc=feed_conc, init_flow_rate=F)

    crn_layer = ReactionNetworkTransitionLayer(reactants, products, k_rxn, d, drive=drive_layer, d_t=d_t)

    # And the equivalent direct implementation of the gray-scott
    gs_layer = models.transition.GrayScottTransitionLayer(d, F, k, d_t=d_t)

    # Iterate them both and compare
    x_vs_t_crn = propagate_x(tiny_numpy_x_0, crn_layer, n_t)
    x_vs_t_gs = propagate_x(tiny_numpy_x_0, gs_layer, n_t)

    # Compare the results using a bit higher tolerance. We know we'll have error because of different ordering and
    # number of fp32 operations, but we're OK with that because it's over a good number of iterations
    assert np.allclose(x_vs_t_crn, x_vs_t_gs, atol=1e-6)

    # And to make sure our tolerances aren't overly generous, run a (short) control with same tolerance
    n_t_short = 10
    gs_layer = models.transition.GrayScottTransitionLayer(d, F, k+1e-4, d_t=d_t)
    x_vs_t_gs = propagate_x(tiny_numpy_x_0, gs_layer, n_t_short)
    assert not np.allclose(x_vs_t_crn[0:n_t_short+1], x_vs_t_gs, atol=1e-6)


def test_gs_crn_vs_gray_scott(tiny_numpy_x_0, gs_k):
    """
    Test the gray scott CRN wrapper against the real deal
    """

    F = .033
    k = gs_k
    d = (.2, .1)

    # Run a good number of iterations for strict comparison but on a small region for speed
    n_t = 128

    crn_layer = models.transition.gray_scott_crn(init_feed_rate=F, init_decay_rate=k, init_diff_coef=d)

    # And the equivalent direct implementation of the gray-scott
    gs_layer = models.transition.GrayScottTransitionLayer(d, F, k)

    # Iterate them both and compare
    x_vs_t_crn = propagate_x(tiny_numpy_x_0, crn_layer, n_t)
    x_vs_t_gs = propagate_x(tiny_numpy_x_0, gs_layer, n_t)

    # Compare the results using a bit higher tolerance. We know we'll have error because of different ordering and
    # number of fp32 operations, but we're OK with that because it's over a good number of iterations
    assert np.allclose(x_vs_t_crn, x_vs_t_gs, atol=1e-6)

    # And to make sure our tolerances aren't overly generous, run a (short) control with same tolerance
    n_t_short = 10
    gs_layer = models.transition.GrayScottTransitionLayer(d, F, k+1e-4)
    x_vs_t_gs = propagate_x(tiny_numpy_x_0, gs_layer, n_t_short)
    assert not np.allclose(x_vs_t_crn[0:n_t_short+1], x_vs_t_gs, atol=1e-6)


@pytest.fixture(params=[.03, .08])
def brusselator_mu(request):
    return request.param


def test_crn_vs_brusselator(tiny_numpy_x_0, brusselator_mu):
    """
    Configure the CRN to be equivalent to the Brusselator model and then compare
    """

    n_t = 128
    d_t = .01

    reactants = np.array([[2, 1, 1],
                          [1, 0, 0]])

    products = np.array([[3, 0, 0],
                         [0, 1, 0]])

    Dx = 1.0
    Dy = 8.0
    A = 4.5
    # Use mu control parameter from Pena & Perez-Garcia PRL 2001
    mu = brusselator_mu
    eta = math.sqrt(1 / Dy)  # Assumes Dx = 1
    Bcrit = (1 + A * eta) ** 2
    B = Bcrit * (mu + 1)
    k_rxn = [1.0, B, 1.0]

    diff_coeff = np.array((Dx, Dy))

    synth_rate = np.array((A, 0.0))
    drive_layer = ConstantSynthesisDriveLayer(synth_rate)

    crn_layer = ReactionNetworkTransitionLayer(reactants, products, k_rxn, diff_coeff, drive=drive_layer, d_t=d_t)

    # And the equivalent direct implementation of the gray-scott
    b_layer = models.transition.BrusselatorTransitionLayer(A, B, Dx, Dy, d_t=d_t)

    # Iterate them both and compare
    x_vs_t_crn = propagate_x(tiny_numpy_x_0, crn_layer, n_t)
    x_vs_t_b = propagate_x(tiny_numpy_x_0, b_layer, n_t)

    # Compare the results using a bit higher tolerance. We know we'll have error because of different ordering and
    # number of fp32 operations, but we're OK with that because it's over a good number of iterations
    assert np.allclose(x_vs_t_crn, x_vs_t_b, atol=1e-6)

    # And to make sure our tolerances aren't overly generous, run a (short) control with same tolerance
    n_t_short = 10
    b_layer = models.transition.BrusselatorTransitionLayer(A, B+1e-4, Dx, Dy, d_t=d_t)
    x_vs_t_b = propagate_x(tiny_numpy_x_0, b_layer, n_t_short)
    assert not np.allclose(x_vs_t_crn[0:n_t_short+1], x_vs_t_b, atol=1e-6)


def test_network_reversibility(reversible_crn_instance):
    """
    Confirm reversibility for reversible predefined networks
    """

    matched_reactions = chem.find_matching_reactions(reversible_crn_instance.reactants,
                                                     reversible_crn_instance.products)

    # Make sure we have matched every reaction to it's reverse
    assert matched_reactions.size == reversible_crn_instance.reactants.shape[1]


def test_add_dx_dt_loss(reversible_crn_instance):
    """
    Basic sanity check of dx/dt penalty term
    """

    loss_wt = 1.0
    reversible_crn_instance.dx_dt_loss_weight = loss_wt

    # Call it to build, add loss term
    x_0 = numpy_x_0_for_crn(reversible_crn_instance)
    x_t = reversible_crn_instance(x_0)

    assert len(reversible_crn_instance.losses) == 1

    dx = reversible_crn_instance.de_center_x(x_t) - reversible_crn_instance.de_center_x(x_0)
    expected_penalty = np.mean(np.abs(dx)) # With max_dx_dt=0...
    assert np.allclose(expected_penalty, reversible_crn_instance.losses[0])

    # Test scaling and threshold
    reversible_crn_instance.dx_dt_loss_weight = loss_wt * 10.0
    x_0 = numpy_x_0_for_crn(reversible_crn_instance)
    x_t = reversible_crn_instance(x_0)

    dx = reversible_crn_instance.de_center_x(x_t) - reversible_crn_instance.de_center_x(x_0)
    expected_penalty = np.mean(np.abs(dx)) * 10.0  # With max_dx_dt=0...
    assert np.allclose(expected_penalty, reversible_crn_instance.losses[0])

    reversible_crn_instance.max_dx_dt = .01
    x_0 = numpy_x_0_for_crn(reversible_crn_instance)
    x_t = reversible_crn_instance(x_0)
    assert expected_penalty > reversible_crn_instance.losses[0]

