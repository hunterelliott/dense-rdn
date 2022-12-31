"""
Accessory code related to chemical reaction network models
"""
from itertools import permutations, combinations, cycle
import numpy as np

import networkx as nx
import tensorflow as tf


def find_matching_reactions(reactants, products):
    """
    Finds reactions which are each other's reverse, taking potential catalysis into account. That is, it matches:
        X + Y -> Z + Y
         with
        Z + Y -> X + Y

    Args:
        reactants: (n_species, n_reactions) matrix of reactant stoichiometry.
        products: (n_species, n_reactions) matrix of product stoichiometry.

    Returns:
        matches: (n_matches, 2) matrix of matched reaction indices. That is
            matches[i, :] = [j, k] is the i-th match and indicates reaction j and reaction k are each other's reverse.

    """

    reactants = np.array(reactants)
    products = np.array(products)

    # Find reactions where reactants = products using broadcasting
    # Creates an (n_species, n_reactions, n_reactions) matrix of comparisons and then does 'all' along the species axis
    matches = np.all(np.expand_dims(reactants, -1) == np.expand_dims(products, 1), axis=0)
    # This gives us True where products = reactants, but we need to be sure reactants = products as well:
    matches = matches & np.transpose(matches)

    assert np.max(np.sum(matches, axis=1)) <= 1, "Some reactions have more than one reverse!"

    # Return as a numpy array of indices, excluding the duplicates
    return np.stack(np.nonzero(np.triu(matches)), -1)


def create_dense_reversible_unimolecular_reaction_network_stoichiometry(n_species):
    """
    Creates the reactant and product stoichiometry matrices for a dense, reversible unimolecular reaction network.
     e.g. one with all possible reversible reactions of the form:
        X <-> Y

    Args:
        n_species: The number of chemical species.

    Returns:
        reactants, products: (n_species, n_reactions) matrix of reaction stoichiometries.
            n_reactions = n_species * (n_species - 1)

    """

    # Get all possible unimolecular X->Y conversions. This line makes me love python but probably makes you hate it.
    reactants, products = zip(*[pair for pair in permutations(list(np.eye(n_species)), 2)])
    reactants = np.transpose(np.stack(reactants, 0))
    products = np.transpose(np.stack(products, 0))

    return reactants, products


def create_dense_reversible_autocatalytic_reaction_network_stoichiometry(n_species):
    """
    Creates the reactant and product stoichiometry matrices for a dense, reversible autocatalytic reaction network with
    all possible reactions of the form:
        X + Y <-> 2Y

    Args:
        n_species: The number of chemical species.

    Returns:
        reactants, products: (n_species, n_reactions) matrix of reaction stoichiometries.
            n_reactions = 2 * n_species * (n_species - 1)

    """

    # First get all possible X->Y conversions
    reactants, products = create_dense_reversible_unimolecular_reaction_network_stoichiometry(n_species)

    # Now make sure that the product is also always a conserved reactant (making the reaction autocatalytic)
    # This converts e.g.  A -> B  into  A + B -> 2B and B -> A into A + B -> 2A
    reactants = reactants + products
    products = products + products

    # Now restore reversibility
    products_rev = np.concatenate([products, reactants], axis=1)
    reactants_rev = np.concatenate([reactants, products], axis=1)

    return reactants_rev, products_rev


def create_dense_mixed_autocatalytic_reaction_network_stoichiometry(n_species, m=2, n=0):
    """
    Creates the reactant and product stoichiometries for a dense, reversible mixed-order autocatalytic reaction network,
    with all reactions of the form:

         X + mY <-> (m+1)Y
        nX +  Y <-> (n+1)X

    m or n may any non-negative integer so that e.g. with m=2, n=0 we get

        X + 2Y <-> 3Y
             Y <-> X

    All possible (X,Y) pairs will be included, but note that if m != n  different species will have different reaction
    sets e.g. one species may have no autocatalytic reactions while another has several.

    Args:
        n_species: The number of chemical species
        m: Y stoichiometry, see above.
        n: X stoichiometry, see above.

    Returns:
        reactants, products: (n_species, n_reactions) matrix of reaction stoichiometries.
            n_reactions = 2 * n_species * (n_species - 1)
    """


    # Get all product-reactant pairs X Y
    all_x, all_y = zip(*[comb for comb in combinations(list(np.eye(n_species)), 2)])
    all_x = np.transpose(np.stack(all_x, 0))
    all_y = np.transpose(np.stack(all_y, 0))

    # Create the m set with Y reactant as autocatalyst with m stoichiometry (if m > 0)
    reactants_m = all_x + m * all_y
    products_m = all_y * (m + 1)

    # Create the n set, with X reactant as autocatalyst with n stoichiometry (if n > 0)
    reactants_n = n * all_x + all_y
    products_n = (n + 1) * all_x

    # Combine them and make the whole system reversible
    reactants = np.concatenate([reactants_m, reactants_n], axis=1)
    products = np.concatenate([products_m, products_n], axis=1)
    products_rev = np.concatenate([products, reactants], axis=1)
    reactants_rev = np.concatenate([reactants, products], axis=1)

    return reactants_rev, products_rev


def create_dense_bimolecular_reaction_network_stoichiometry(n_species):
    """
    Creates the reactant and product stoichiometry matrices for a dense, reversible reaction network with all possible
    reactions of the form:
        X + Y <-> 2Z

    Args:
        n_species: The number of chemical species ( >=3 )

    Returns:
        reactants, products: (n_species, n_reactions) matrix of reaction stoichiometries.
            n_reactions = n_species * (n_species - 1) * (n_species - 2)

    """
    assert n_species >= 3, "This network topology requires at least 3 chemical species!"

    # Get all pairs of reactants X + Y
    reactants = np.transpose(np.stack([comb[0] + comb[1] for comb in combinations(list(np.eye(n_species)), 2)], axis=0))

    # The non-reactants are the possible products
    n_prod_poss = n_species - 2
    rxn_ind, prod_poss = np.nonzero(np.transpose(1 - reactants))

    # Replicate the product array with all the possible products for each set of reactants
    products = np.zeros((n_prod_poss,) + reactants.shape)
    products[np.tile(range(n_prod_poss),reactants.shape[1]), prod_poss, rxn_ind] = 2
    products = np.concatenate(list(products), axis=1)

    # Replicate the reactants to match
    reactants = np.concatenate([reactants for _ in range(n_prod_poss)], axis=1)

    # Finally, make the whole thing reversible
    products_rev = np.concatenate([products, reactants], axis=1)
    reactants_rev = np.concatenate([reactants, products], axis=1)

    return reactants_rev, products_rev


def create_dense_2_species_third_order_reaction_network_stoichiometry(n_species):
    """
    Creates the reactant and product stoichiometry matrices for a dense, reversible reaction network with
    reactions of the form:
        X + 2Y <-> 3Z

    All pairs of species (X, Y) will react, but for a given pair only one will be second-order (and that same species
    may instead be first order in a different pairing).

    Args:
        n_species: The number of chemical species ( >=3 )

    Returns:
        reactants, products: (n_species, n_reactions) matrix of reaction stoichiometries.
            n_reactions = n_species * (n_species - 1) * (n_species - 2)

    """
    assert n_species >= 3, "This network topology requires at least 3 chemical species!"

    # Get all pairs of reactants X + 2Y
    reactants = np.transpose(np.stack([comb[0] + comb[1]*2 for comb in combinations(list(np.eye(n_species)), 2)], axis=0))

    # The non-reactants are the possible products
    n_prod_poss = n_species - 2
    rxn_ind, prod_poss = np.nonzero(np.transpose(np.logical_not(reactants>0)))

    # Replicate the product array with all the possible products for each set of reactants
    products = np.zeros((n_prod_poss,) + reactants.shape)
    products[np.tile(range(n_prod_poss),reactants.shape[1]), prod_poss, rxn_ind] = 3
    products = np.concatenate(list(products), axis=1)

    # Replicate the reactants to match
    reactants = np.concatenate([reactants for _ in range(n_prod_poss)], axis=1)

    # Finally, make the whole thing reversible
    products_rev = np.concatenate([products, reactants], axis=1)
    reactants_rev = np.concatenate([reactants, products], axis=1)

    return reactants_rev, products_rev


def create_dense_2_species_mixed_order_reaction_network_stoichiometry(n_species, stoich_list=(1, 2)):
    """
    Creates the reactant and product stoichiometry matrices for a dense, reversible reaction network with
    reactions of the form:
        X + aY <-> (a+1)Z

    Where a is cycled through the options in the input list.  All pairs of species (X, Y) will react, but for a given
    pair only one will be >1st-order (and that same species may instead be first order in a different pairing).

    Args:
        n_species: The number of chemical species ( >=3 )
        stoich_list: iterable which will be cycled through to produce the stoichiometry for the Y species in each
            reaction. e.g. if (1,2) is input the Y stoichiometries will be 1,2,1,2,...

    Returns:
        reactants, products: (n_species, n_reactions) matrix of reaction stoichiometries.
            n_reactions = n_species * (n_species - 1) * (n_species - 2)

    """
    assert n_species >= 3, "This network topology requires at least 3 chemical species!"
    assert all([s > 0 for s in stoich_list]), "All Y stoichiometries must be >0"

    # Get all pairs of reactants X + 2Y
    y_stoich = cycle(stoich_list)
    reactants = np.transpose(np.stack([comb[0] + comb[1]*next(y_stoich)
                                       for comb in combinations(list(np.eye(n_species)), 2)], axis=0))

    # The non-reactants are the possible products
    n_prod_poss = n_species - 2
    rxn_ind, prod_poss = np.nonzero(np.transpose(np.logical_not(reactants>0)))

    # Replicate the product array with all the possible products for each set of reactants
    products = np.zeros((n_prod_poss,) + reactants.shape)
    products[np.tile(range(n_prod_poss), reactants.shape[1]), prod_poss, rxn_ind] = np.sum(reactants, axis=0)[rxn_ind]
    products = np.concatenate(list(products), axis=1)

    # Replicate the reactants to match
    reactants = np.concatenate([reactants for _ in range(n_prod_poss)], axis=1)

    # Finally, make the whole thing reversible
    products_rev = np.concatenate([products, reactants], axis=1)
    reactants_rev = np.concatenate([reactants, products], axis=1)

    return reactants_rev, products_rev


def create_dense_kitchen_sink_reaction_network_stoichiometry(n_species):
    """
    Creates the reactant and product stoichiometry matrices for a dense reversible reaction network with reactions of
    three types:

        X +  Y <-> 2Z
             Y <-> X
        X + 2Y <-> 3Y

    with all possible (X, Y, Z) combinations (with some restrictions on the autocatalytic reactions).

    Args:
        n_species: The number of chemical species ( >= 3 )

    Returns:
        reactants, products: (n_species, n_reactions) matrix of reaction stoichiometries.
            n_reactions = n_species^2 * (n_species - 1)
    """

    # With n=0 the mixed autocat stoich also gives us the unimolecular X<->Y reactions.
    reactants_1, products_1 = create_dense_mixed_autocatalytic_reaction_network_stoichiometry(n_species, m=2, n=0)
    reactants_2, products_2 = create_dense_bimolecular_reaction_network_stoichiometry(n_species)

    reactants = np.concatenate([reactants_1, reactants_2], axis=1)
    products = np.concatenate([products_1, products_2], axis=1)

    return reactants, products


def get_reaction_network_rates(c, reactants, rate_const):
    """
    Calculate local reaction rates for a chemical reaction network model
    Args:
        c: (batch, height, width, n_species) array/tensor of local concentrations in mol/L
        reactants: (n_species, n_reactions) matrix of reactant stoichiometries
        rate_const: (n_reactions) vector of reaction rate constants

    Returns:
        rates: (batch, height, width, n_reaction) array of local reaction rates.
    """

    n_species, n_reactions = reactants.shape
    # Raise each concentration to the power dictated by it's stoichiometry, with the magic of broadcasting and
    # avoiding NaN gradients from 0**0
    # Gives a (batch, h, w, n_species, n_reactions)
    x_pows = tf.expand_dims(tf.where(tf.equal(c, 0.), 1e-10, c), -1) ** \
             tf.reshape(reactants, (1, 1, 1, n_species, n_reactions))

    # Take the appropriate (possibly exponentiated) concentration products and multiply by rate constant
    # This gives us a (batch, h, w, n_reaction) tensor of reaction rates.
    return tf.reduce_prod(x_pows, axis=-2) * tf.reshape(rate_const, (1, 1, 1, n_reactions))


def get_complexes(reactants, products):
    """
    Returns the "complexes" in the input chemical reaction network, where complexes are defined in CRN terminology as
    any unique species stoichiometry involved in any reaction, e.g. for
        A + 2B -> C
        C + A -> B
    The complexes would be A + 2B, C, C + A, B

    Args:
        reactants: (n_species, n_reactions) matrix of reactant stoichiometries
        products: (n_species, n_reactions) matrix of product stoichiometries

    Returns:
        complexes: (n_species, n_complexes) matrix of complexes (with stoichiometry)

    """

    return np.unique(np.concatenate([reactants, products], axis=1), axis=1)


def get_complex_reaction_edges(reactants, products, complexes):
    """
    Returns the "reaction edges" for the input complexes given the reaction network specified by (reactants, products)
    where a reaction edge indicates that one complex reacts to form another.

    Args:
        reactants: (n_species, n_reactions) matrix of reactant stoichiometries
        products: (n_species, n_reactions) matrix of product stoichiometries
        complexes: (n_species, n_complexes) matrix of complexes (with stoichiometry)

    Returns:
        i_reactant_complex, i_product_complex: vectors which index into complexes, such that
            complexes[:,i_reactant_complex[j]] -> complexes[:,i_product_complex[j]]
            is the j-th reaction edge.

    """

    # Add some dimensions so we can use broadcasting to compare
    complexes = np.expand_dims(complexes, axis=1)
    reactants = np.expand_dims(reactants, axis=-1)
    products = np.expand_dims(products, axis=-1)

    # Now match products and reactants to complexes
    # This creates an n_reaction x n_complex binary comparison matrix then argmaxes that to complex indices
    i_reactant_complexes = np.argmax(np.all(complexes == reactants, axis=0), axis=1)
    i_product_complexes = np.argmax(np.all(complexes == products, axis=0), axis=1)

    return i_reactant_complexes, i_product_complexes


def get_species_human_names(n_species):
    """
    Returns a list of strings for assigning more human-friendly names to species
    Args:
        n_species: number of chemical species

    Returns:
        species_names = length n_species list of names ['A', 'B',...]

    """
    return [chr(ord('A') + i) for i in range(n_species)]


def get_complex_human_names(complexes):
    """
    Returns human-friendly names for each of the complexes.
    Args:
        complexes: (n_species, n_complexes) matrix of complexes (with stoichiometry)

    Returns:
        complex_names: length n_complexes list of names.

    """

    assert np.all(complexes.astype(np.int32) == complexes), "This function doesn't support non-integer stoichiometries!"
    complexes = complexes.astype(np.int32)

    species_names = get_species_human_names(complexes.shape[0])
    names = []
    for i_complex in range(complexes.shape[1]):
        name = ''
        i_species = np.nonzero(complexes[:, i_complex])[0]
        for n, i, in enumerate(i_species):
            if complexes[i, i_complex] > 1:  # By convention we omit stoichiometric coefficients of 1
                name = name + str(int(complexes[i, i_complex]))
            name = name + species_names[i]
            if n < len(i_species) - 1:
                name = name + ' + '
        names.append(name)

    return names


def get_crn_complex_graph(crn):
    """
    Returns a networkx directed graph corresponding to the input chemical reaction network (CRN).This graph has
    complexes (e.g. A + 2B) as nodes, and reactions as edges with edge weights set to the rate constants.

    Args:
        crn: a ReactionNetworkTransitionLayer describing the CRN.

    Returns:
        graph: A NetworkX DiGraph object with n_complexes nodes and n_reactions directed edges.

    """

    # Get the complexes (nodes) and edges (reactions) from the CRN
    complexes = get_complexes(crn.reactants, crn.products)
    i_rxt_edges, i_prod_edges = get_complex_reaction_edges(crn.reactants, crn.products, complexes)

    # Put human-friendly names on them and weights equal to rate constants
    complex_names = get_complex_human_names(complexes)
    named_edges = [(complex_names[i_rxt], complex_names[i_prod]) for (i_rxt, i_prod) in zip(i_rxt_edges, i_prod_edges)]
    edges_and_weights = [edge + ({'weight': crn.rate_const[i_edge]},) for (i_edge, edge) in enumerate(named_edges)]

    # Now create the NetworkX directed graph
    graph = nx.DiGraph()
    graph.add_nodes_from(complex_names)
    graph.add_edges_from(edges_and_weights)

    return graph
