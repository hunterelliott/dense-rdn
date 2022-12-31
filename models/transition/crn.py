"""
Generic chemical reaction network reaction-diffusion models (and derivatives thereof).
"""
import logging
import numpy as np
from scipy.constants import Boltzmann as k_b

import tensorflow as tf
from tensorflow.python.keras.constraints import NonNeg
from tensorflow.python.keras.initializers.initializers_v2 import Constant
from tensorflow.python.keras.layers import serialize, deserialize, Lambda

from models.transition.base import TransitionLayer
from ops import pad_toroidal, laplacian_kernel, MinMaxValueConstraint
from models.drive import FlowDriveLayer, ZeroDriveLayer
from therm import diffusion_entropy_rate, reaction_entropy_rate
import chem


@tf.keras.utils.register_keras_serializable(package="Iteration")
class ReactionNetworkTransitionLayer(TransitionLayer):
    """
    A transition layer for reaction-diffusion modelling of arbitrary chemical reaction networks (CRNs).
    """
    def __init__(self, reactants, products, init_rate_const=None, init_diff_coeffs=None,
                 fit_rate_const=False, fit_diff_coeffs=False, drive=ZeroDriveLayer(),
                 diffusion_entropy_loss_weight=0.0, reaction_entropy_loss_weight=0.0,
                 dx_dt_loss_weight=0.0, max_dx_dt=0.0, **kwargs):
        """
        Args:
            reactants: (n_species, n_reactions) matrix of reactant stoichiometries
            products: (n_species, n_reactions) matrix of product stoichiometries
            init_rate_const: (n_reactions) vector of initial values for rate constants for each reaction
            init_diff_coeffs: (n_species) vector of initial values for diffusion coefficients
            fit_rate_const: If True, rate constants will be a fittable (trainable) parameter.
            fit_diff_coeffs: If True, diffusion coefficients  will be a fittable (trainable) parameter.
            drive: A function / Layer specifying a drive for modeling non-equilibrium systems,
                of the form dx_dt=f(x_t, i_t). Should be serializable.
            diffusion_entropy_loss_weight: If non-zero, diffusion_entropy_rate loss term will be added using add_loss.
            reaction_entropy_loss_weight: If non-zero, reaction_entropy_rate loss term will be added using add_loss.
            dx_dt_loss_weight: If non-zero, a loss term will be applied for dx/dt values above max_dx_dt
            max_dx_dt: Threshold for dx_dt_loss (if weight is non-zero).
            **kwargs:
        """

        super(ReactionNetworkTransitionLayer, self).__init__(**kwargs)

        self.reactants = np.array(reactants, dtype=np.float32)
        self.products = np.array(products, dtype=np.float32)
        self.init_rate_const = init_rate_const
        self.init_diff_coeffs = init_diff_coeffs
        self.fit_rate_const = fit_rate_const
        self.fit_diff_coeffs = fit_diff_coeffs
        self.drive = drive

        self.diffusion_entropy_loss_weight = diffusion_entropy_loss_weight
        self.reaction_entropy_loss_weight = reaction_entropy_loss_weight
        self.dx_dt_loss_weight = dx_dt_loss_weight
        self.max_dx_dt = max_dx_dt

        # Derived parameters and input validation
        self.s = self.products - self.reactants  # The "stoichiometry matrix" in CRN terminology
        self.n_species = self.reactants.shape[0]
        self.n_reactions = self.reactants.shape[1]
        self.reaction_matches = chem.find_matching_reactions(self.reactants, self.products)
        self.is_reversible = self.reaction_matches.shape[0] == int(self.n_reactions/2)

        if self.init_rate_const is None:
            self.init_rate_const = np.ones(self.n_reactions) * .1
        if self.init_diff_coeffs is None:
            self.init_diff_coeffs = np.ones(self.n_species) * .1

        assert self.products.shape == (self.n_species, self.n_reactions)
        assert len(self.init_rate_const) == self.n_reactions
        assert len(self.init_diff_coeffs) == self.n_species
        if self.reaction_entropy_loss_weight != 0 or self.diffusion_entropy_loss_weight != 0:
            assert self.is_reversible, "Irreversible reaction entropy losses not yet supported!"

    def build(self, input_shape):
        self.rate_const = self.add_weight(shape=self.n_reactions, initializer=Constant(value=self.init_rate_const),
                                          name='rate_const', constraint=NonNeg(), trainable=self.fit_rate_const)
        self.diff_coeffs = self.add_weight(shape=self.n_species, initializer=Constant(value=self.init_diff_coeffs),
                                           name='diff_coeffs',
                                           constraint=MinMaxValueConstraint(min_value=.05, max_value=.2),  # Found to be numerically stable
                                           trainable=self.fit_diff_coeffs)

    def call(self, x_t, i_t=0, seed=42, **kwargs):

        x_t = self.de_center_x(x_t)

        # Ge the (batch, h, w, n_reaction) tensor of reaction rates.
        rates = chem.get_reaction_network_rates(x_t, self.reactants, self.rate_const)

        # We use the stoichiometry matrix to get the per-species d/dt (in mol/s) due to reactions
        # This syntax lets us do a matrix multiplication over the last dimensions, broadcasting over the rest.
        # This gives a (batch, h, w, n_species) tensor of concentration derivatives
        dx_dt = tf.squeeze(tf.einsum('ij,...jk->...ik', self.s, tf.expand_dims(rates, -1)))

        # Now calculate the time derivative component due to diffusion
        # We use a discrete approximation of the laplacian
        del2_x = tf.nn.depthwise_conv2d(pad_toroidal(x_t, 1), laplacian_kernel(self.n_species), [1, 1, 1, 1], 'VALID')
        dx_dt = dx_dt + del2_x * tf.reshape(self.diff_coeffs, (1, 1, 1, self.n_species))

        # Include drive term
        dx_dt = dx_dt + self.drive(x_t, i_t)

        if self.is_reversible:
            # Handle the thermodynamic losses, applied to x_t before integration.
            diff_entropy_rate = tf.reduce_mean(diffusion_entropy_rate(x_t, self.diff_coeffs)) * self.d_t
            self.add_metric(diff_entropy_rate, name='diffusion_entropy_rate')
            if not self.diffusion_entropy_loss_weight == 0:
                self.add_loss(self.scale_diffusion_loss(diff_entropy_rate))

            rxn_entropy_rate = tf.reduce_mean(reaction_entropy_rate(tf.gather(rates, self.reaction_matches[:, 0], axis=3),
                                                                    tf.gather(rates, self.reaction_matches[:, 1], axis=3)))
            self.add_metric(rxn_entropy_rate, name='reaction_entropy_rate')
            if not self.reaction_entropy_loss_weight == 0:
                self.add_loss(self.scale_reaction_loss(rxn_entropy_rate))

        # Finally, perform simple Euler integration
        x_t = x_t + dx_dt * self.d_t

        self.add_metric(tf.reduce_mean(tf.abs(dx_dt)), name="mean_abs_dxdt")
        self.add_metric(tf.reduce_max(tf.abs(dx_dt)), name="max_abs_dxdt")
        if self.dx_dt_loss_weight != 0:
            dx_penalty = tf.reduce_mean(tf.maximum(tf.abs(dx_dt)-self.max_dx_dt, [0.0]))
            self.add_loss(self.scale_dx_dt_loss(dx_penalty))
            self.add_metric(dx_penalty, name='dxdt_penalty')

        # Clip any negative concentrations resulting from integration error, as well as enforcing a concentration cap
        x_t = tf.clip_by_value(x_t, 0.0, 1e1)

        return self.center_x(x_t)

    def scale_dx_dt_loss(self, dx_penalty):
        return tf.multiply(dx_penalty, self.dx_dt_loss_weight, name='dx_dt_penalty_loss')

    def scale_diffusion_loss(self, diff_entropy_rate):
        # We allow user-specified weighting and remove the factor of kb to bring it into order 1 magnitude.
        return tf.math.multiply(tf.exp(-diff_entropy_rate * 1/k_b), self.diffusion_entropy_loss_weight,
                                name='diffusion_entropy_rate_loss')

    def scale_reaction_loss(self, rxn_entropy_rate):
        return tf.math.multiply(tf.exp(-rxn_entropy_rate * 1/k_b), self.reaction_entropy_loss_weight,
                                name='reaction_entropy_rate_loss')

    def get_config(self):
        config = super(ReactionNetworkTransitionLayer, self).get_config()
        config.update({'reactants': self.reactants,
                       'products': self.products,
                       'init_rate_const': self.init_rate_const,
                       'init_diff_coeffs': self.init_diff_coeffs,
                       'fit_rate_const': self.fit_rate_const,
                       'fit_diff_coeffs': self.fit_diff_coeffs,
                       'drive': serialize(self.drive),
                       'diffusion_entropy_loss_weight': self.diffusion_entropy_loss_weight,
                       'reaction_entropy_loss_weight': self.reaction_entropy_loss_weight,
                       'dx_dt_loss_weight': self.dx_dt_loss_weight,
                       'max_dx_dt': self.max_dx_dt})
        return config

    @classmethod
    def from_config(cls, config):
        config['drive'] = deserialize(config['drive'])
        return cls(**config)

    def log_free_parameters(self):
        logging.info("Diffusion coefficients: {}".format(self.diff_coeffs.numpy()))
        logging.info("Reaction rate constants: {}".format(self.rate_const.numpy()))
        self.drive.log_free_parameters()


def coupled_gray_scott_crn(feed_rate=.03, decay_rates=(.06, .056), coupling_rate=1.5e-3, diff_coeffs=(.2, .1, .16, .1),
                           fit_params=False, **kwargs):
    """
    Builds a 4-species "coupled" gray scott model CRN, with two feed species, two autocatalysts and a reaction
    converting one autocatalyst into the other:

        U + 2V -> 3V
        V -> Decay
        W + 2X -> 3X
        X -> Decay
        V -> X

    Args:
        feed_rate: Feed rate for FlowDriveLayer
        decay_rates: tuple of decay rates for the two autocatalysts
        coupling_rate: rate constant for the V->X coupling reaction.
        diff_coeffs: length-4 vector of diffusion constants.
        fit_params: If True, then diffusion coefficients and reaction rates will be trainable parameters.

    Returns:
        model: a ReactionNetworkLayer instance.
    """

    reactants = [[1, 0, 0, 0, 0],
                 [2, 1, 0, 0, 1],
                 [0, 0, 1, 0, 0],
                 [0, 0, 2, 1, 0]]

    products = [[0, 0, 0, 0, 0],
                [3, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 3, 0, 1]]

    rate_const = [1.0, decay_rates[0], 1.0, decay_rates[1], coupling_rate]

    feed_conc = [1.0, 0.0, 1.0, 0.0]

    drive = FlowDriveLayer(init_feed_conc=feed_conc, init_flow_rate=feed_rate)

    return ReactionNetworkTransitionLayer(reactants=reactants, products=products, init_rate_const=rate_const,
                                          init_diff_coeffs=diff_coeffs, drive=drive,
                                          fit_diff_coeffs=fit_params, fit_rate_const=fit_params, **kwargs)


def reversible_coupled_gray_scott_crn(feed_rate=.03, decay_rates=(.06, .056), coupling_rate=1.5e-3,
                                      diff_coeffs=(.2, .1, .16, .1, .05), reverse_rate=1e-3,
                                      drive_class=FlowDriveLayer, drive_kwargs=None, **kwargs):
    """
    Builds a 4-species "coupled" reversible gray scott model CRN, with two feed species, two autocatalysts and a
    reaction converting one autocatalyst into the other. The inert version of the autocatalysts is shared:

        U + 2V <-> 3V
        V <-> P
        W + 2X <-> 3X
        X <-> P
        V <-> X

    Args:
        feed_rate: Feed rate for FlowDriveLayer
        decay_rates: tuple of decay rates for the two autocatalysts
        coupling_rate: rate constant for the V->X coupling reaction.
        diff_coeffs: length-4 vector of diffusion constants.
        reverse_rate: The rate constants for the reverse reactions.

    Returns:
        model: a ReactionNetworkLayer instance.
    """
    if drive_kwargs is None:
        drive_kwargs = {}

    reactants = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [2, 1, 0, 0, 1, 3, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 2, 1, 0, 0, 0, 3, 0, 1],
                 [0, 0, 0, 0, 0, 0, 1, 0, 1, 0]]

    products = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [3, 0, 0, 0, 0, 2, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 3, 0, 1, 0, 0, 2, 1, 0],
                [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]]

    rate_const = [1.0, decay_rates[0], 1.0, decay_rates[1], coupling_rate] + [reverse_rate for _ in range(5)]

    feed_conc = [1.0, 0.0, 1.0, 0.0, 0.0]

    if 'init_feed_conc' not in drive_kwargs:
        # And we initialize such that less autocatalytic species are fed at higher concentrations
        drive_kwargs['init_feed_conc'] = feed_conc
    if 'init_flow_rate' not in drive_kwargs:
        drive_kwargs['init_flow_rate'] = feed_rate

    drive = drive_class(**drive_kwargs)

    return ReactionNetworkTransitionLayer(reactants=reactants, products=products, init_rate_const=rate_const,
                                          init_diff_coeffs=diff_coeffs, drive=drive, **kwargs)


def gray_scott_crn(init_feed_rate=.03, init_decay_rate=.06, init_diff_coef=(.2, .1), **kwargs):
    """
    Builds the gray scott model as a CRN instance purely for testing purposes
    """

    reactants = np.array([[1, 0],
                          [2, 1]])

    products = np.array([[0, 0],
                         [3, 0]])

    # The non-dimensionalized gray scott assumes k=1 for autocatalytic reaction
    k_rxn = [1.0, init_decay_rate]
    # It also assumes only U is fed and at a concentration of 1
    feed_conc = (1.0, 0.0)
    drive_layer = FlowDriveLayer(init_feed_conc=feed_conc, init_flow_rate=init_feed_rate)

    return ReactionNetworkTransitionLayer(reactants, products, k_rxn, init_diff_coef, drive=drive_layer, **kwargs)


def reversible_gray_scott_crn(init_feed_rate=.03, init_decay_rate=.06, init_diff_coef=(.2, .1, .05),
                              reverse_rate=1e-3, drive=None, **kwargs):
    """
    Builds a reversible version of the gray scott model as a CRN, again for testing.
    """

    reactants = np.array([[1, 0, 0, 0],
                          [2, 1, 3, 0],
                          [0, 0, 0, 1]])

    products = np.array([[0, 0, 1, 0],
                         [3, 0, 2, 1],
                         [0, 1, 0, 0]])

    # The non-dimensionalized gray scott assumes k=1 for autocatalytic reaction
    k_rxn = [1.0, init_decay_rate, reverse_rate, reverse_rate]
    # It also assumes only U is fed and at a concentration of 1
    feed_conc = (1.0, 0.0, 0.0)

    if drive is None:
        drive = FlowDriveLayer(init_feed_conc=feed_conc, init_flow_rate=init_feed_rate)

    return ReactionNetworkTransitionLayer(reactants, products, k_rxn, init_diff_coef, drive=drive, **kwargs)


def dense_crn(n_species, stoich_name, max_init_rate_const=1e-3, **kwargs):
    """
    Builds a dense reaction network CRN using the specified stoichiometry creation method.
    Args:
        n_species: The number of chemical species.
        stoich_name: Name of function in the chem module to use to create reaction stoichiometry.
        max_init_rate_const: Initial rate constants will be drawn from uniform(0, max_init_rate_const)
        **kwargs: Additional arguments ot pass to ReactionNetworkTransitionLayer

    Returns:
        model: A ReactionNetworkTranstionLayer model.

    """
    reactants, products = getattr(chem, stoich_name)(n_species)

    n_reactions = reactants.shape[1]
    init_rate_const = np.random.uniform(0, max_init_rate_const, n_reactions)
    init_diff_coeffs = np.random.uniform(0.05, .2, n_species)

    return ReactionNetworkTransitionLayer(reactants, products, init_rate_const, init_diff_coeffs, **kwargs)


def dense_gs_crn(n_species, drive_class=FlowDriveLayer, drive_kwargs=None, **kwargs):
    """
    Creates a dense mixed order autocatalytic CRN with stoichiometry and initialization that is analagous to an
     n-species version of the Gray Scott system.
    Args:
        n_species: The number of chemical species.

    Returns:
        model: A ReactionNetworkTransitionLayer model.
    """
    if drive_kwargs is None:
        drive_kwargs = {}

    m = 2
    reactants, products = chem.create_dense_mixed_autocatalytic_reaction_network_stoichiometry(n_species, m=m, n=0)

    # Number of species pairs
    n_pairs = int(n_species * (n_species - 1) / 2)
    n_reactions = reactants.shape[1]

    init_rate_const = np.zeros(n_reactions)
    # # Initialize all the autocatalysis reactions rates to 1.0
    # init_rate_const[0:n_pairs] = 1.0
    i_first_autocat = np.argmax(products == (m + 1), axis=1)
    # Initialize the first autocatalytic reaction for each species to rate 1.0
    # These will also be reactions which all draw from species 0 which limits runaway initial reaction rates
    init_rate_const[i_first_autocat] = 1.0

    # Initialize the unimolecular interconversion rates to a small value
    init_rate_const[n_pairs:2*n_pairs] = 1e-3
    # And the reverse reaction rates to near-zero
    init_rate_const[2*n_pairs:] = 1e-3
    # And break any symmetry with small random noise
    init_rate_const = init_rate_const + np.random.uniform(-1e-4, 1e-4, init_rate_const.shape)

    init_diff_coeffs = np.random.uniform(0.1, .2, n_species)

    if 'init_feed_conc' not in drive_kwargs:
        # And we initialize such that less autocatalytic species are fed at higher concentrations
        drive_kwargs['init_feed_conc'] = 1 / (2*m)**np.sum(reactants[:, 0:n_pairs] == m, axis=1)
    if 'init_flow_rate' not in drive_kwargs:
        drive_kwargs['init_flow_rate'] = .03

    drive = drive_class(**drive_kwargs)

    return ReactionNetworkTransitionLayer(reactants=reactants, products=products, init_rate_const=init_rate_const,
                                          init_diff_coeffs=init_diff_coeffs, drive=drive, **kwargs)


def dense_kitchen_sink_crn(n_species, drive_class=FlowDriveLayer, drive_kwargs=None, **kwargs):
    """
    Creates a "kitchen sink" CRN with stoichiometry and initialization for the autocatalytic and unimolecular portion
     analagous to an n-species version of the Gray Scott system, and bimolecular reaction initialized near-zero.
    Args:
        n_species: The number of chemical species.

    Returns:
        model: A ReactionNetworkTransitionLayer model.
    """
    if drive_kwargs is None:
        drive_kwargs = {}
    m = 2
    reactants, products = chem.create_dense_kitchen_sink_reaction_network_stoichiometry(n_species)

    n_gs_rxns = 2 * n_species * (n_species - 1) # Number of reactions from the gay-scott-like set.

    # Number of species pairs
    n_pairs = int(n_species * (n_species - 1) / 2)
    n_reactions = reactants.shape[1]

    init_rate_const = np.zeros(n_reactions)
    # # Initialize all the autocatalysis reactions rates to 1.0
    # init_rate_const[0:n_pairs] = 1.0
    i_first_autocat = np.argmax(products == (m + 1), axis=1)
    # Initialize the first autocatalytic reaction for each species to rate 1.0
    # These will also be reactions which all draw from species 0 which limits runaway initial reaction rates
    init_rate_const[i_first_autocat] = 1.0

    # Initialize the unimolecular interconversion rates to a small value
    init_rate_const[n_pairs:2*n_pairs] = 1e-3
    # And the reverse reaction rates and bimolecular rates to near-zero
    init_rate_const[2*n_pairs:] = 1e-3
    # And break any symmetry with small random noise
    init_rate_const = init_rate_const + np.random.uniform(-1e-4, 1e-4, init_rate_const.shape)

    init_diff_coeffs = np.random.uniform(0.1, .2, n_species)

    if 'init_feed_conc' not in drive_kwargs:
        # And we initialize such that less autocatalytic species are fed at higher concentrations
        drive_kwargs['init_feed_conc'] = 1 / (2*m)**np.sum(reactants[:, 0:n_pairs] == m, axis=1)
    if 'init_flow_rate' not in drive_kwargs:
        drive_kwargs['init_flow_rate'] = .03

    drive = drive_class(**drive_kwargs)

    return ReactionNetworkTransitionLayer(reactants=reactants, products=products, init_rate_const=init_rate_const,
                                          init_diff_coeffs=init_diff_coeffs, drive=drive, **kwargs)