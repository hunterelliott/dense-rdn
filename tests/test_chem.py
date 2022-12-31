"""
Tests for CRN accessory functions
"""


import numpy as np
import pytest

import chem
from models.transition.crn import ReactionNetworkTransitionLayer


def get_test_crns(ind):
    # Define test reaction networks separately so they're not all in a huge fixture and so
    # we can mix and match for different tests

    if ind == 0:
        # Reverisble A <--> B catalyzed by C, e.g.:
        # A + C -> B + C and B + C -> A + C
        reactants = [[1, 0],
                     [0, 1],
                     [1, 1]]

        products = [[0, 1],
                    [1, 0],
                    [1, 1]]

    if ind == 1:
        # Reversible 2A <-> B and B<->2C
        reactants = [[2, 0, 0, 0],
                     [0, 1, 1, 0],
                     [0, 0, 0, 2]]
        products = [[0, 2, 0, 0],
                    [1, 0, 0, 1],
                    [0, 0, 2, 0]]

    if ind == 2:
        # Irreversible A + B -> C and C->D
        reactants = [[1, 0],
                     [1, 0],
                     [0, 1],
                     [0, 0]]
        products = [[0, 0],
                    [0, 0],
                    [1, 0],
                    [0, 1]]

    if ind == 3:
        # Mixed, reversible A + B -> C and irreversible C->D
        reactants = [[1, 0, 0],
                     [1, 0, 0],
                     [0, 1, 1],
                     [0, 0, 0]]
        products = [[0, 0, 1],
                    [0, 0, 1],
                    [1, 0, 0],
                    [0, 1, 0]]

    if ind == 4:
        # Goofy - Reversible 2A <-> B and B<->2C with duplicate 2C-> B
        reactants = [[2, 0, 0, 0, 0],
                     [0, 1, 1, 0, 0],
                     [0, 0, 0, 2, 2]]
        products = [[0, 2, 0, 0, 0],
                    [1, 0, 0, 1, 1],
                    [0, 0, 2, 0, 0]]

    return reactants, products


@pytest.fixture(params=[[0, [0, 1]],
                        [1, [[0, 1], [2, 3]]],
                        [3, [0, 2]]])
def reaction_net_reversible_stoich(request):
    return get_test_crns(request.param[0]), np.array(request.param[1])


@pytest.fixture(params=[2])
def reaction_net_irreversible_stoich(request):
    return get_test_crns(request.param)


@pytest.fixture(params=[4])
def reaction_net_goofy_stoich(request):
    return get_test_crns(request.param)


@pytest.fixture(params=[4, 3, 5])
def n_species(request):
    return request.param


def test_match_reactions_reversible(reaction_net_reversible_stoich):
    (reactants, products), expected_matches = reaction_net_reversible_stoich

    matches = chem.find_matching_reactions(reactants, products)

    assert np.all(matches == expected_matches)
    assert matches.size == np.array(expected_matches).size  # Because np.all does broadcasting...


def test_match_reactions_irreversible(reaction_net_irreversible_stoich):
    reactants, products = reaction_net_irreversible_stoich

    matches = chem.find_matching_reactions(reactants, products)

    assert np.shape(matches) == (0, 2)


def test_match_reactions_broken(reaction_net_goofy_stoich):
    reactants, products = reaction_net_goofy_stoich

    with pytest.raises(AssertionError):
        matches = chem.find_matching_reactions(reactants, products)


def test_dense_uni_rxn_stoich(n_species):

    reactants, products = chem.create_dense_reversible_unimolecular_reaction_network_stoichiometry(n_species)

    n_reactions = n_species * (n_species - 1)
    assert reactants.shape[1] == n_reactions

    matches = chem.find_matching_reactions(reactants, products)

    assert matches.shape[0] == n_reactions / 2


def test_dense_autocat_rxn_stoich(n_species):

    reactants, products = chem.create_dense_reversible_autocatalytic_reaction_network_stoichiometry(n_species)

    n_reactions = 2 * n_species * (n_species - 1)
    assert reactants.shape[1] == n_reactions

    matches = chem.find_matching_reactions(reactants, products)

    assert matches.shape[0] == n_reactions / 2


def test_dense_bimol_rxn_stoich(n_species):

    reactants, products = chem.create_dense_bimolecular_reaction_network_stoichiometry(n_species)

    n_reactions = n_species * (n_species - 1) * (n_species - 2)
    assert reactants.shape[1] == n_reactions

    matches = chem.find_matching_reactions(reactants, products)

    assert matches.shape[0] == n_reactions / 2

    # Make sure a reactant is never a product
    assert not np.any(np.logical_and(products > 0, reactants > 0))

    # The # per reaction are right
    assert np.all(np.sum(reactants, axis=0) == 2)
    assert np.all(np.sum(products, axis=0) == 2)

    # and that we're symmetric..
    assert len(np.unique(np.sum(reactants, axis=1))) == 1
    assert np.all(np.unique(np.sum(reactants, axis=1)) == np.unique(np.sum(products, axis=1)))


def test_dense_2spec_third_order_rxn_stoich(n_species):

    reactants, products = chem.create_dense_2_species_third_order_reaction_network_stoichiometry(n_species)

    n_reactions = n_species * (n_species - 1) * (n_species - 2)
    assert reactants.shape[1] == n_reactions

    matches = chem.find_matching_reactions(reactants, products)

    assert matches.shape[0] == n_reactions / 2

    # Make sure a reactant is never a product
    assert not np.any(np.logical_and(products > 0, reactants > 0))

    # Make sure the stoichiometries are conserved
    assert np.all(np.sum(reactants, axis=0) == np.sum(products, axis=0))

    # And they're all 3
    assert np.unique(np.sum(reactants, axis=0)) == np.unique(np.sum(products, axis=0)) == 3

    # And that each is in the same number of reactions. I don't know, I'm tired.
    assert np.all(np.unique(np.sum(reactants > 0, axis=1)) == np.unique(np.sum(products > 0, axis=1)))


@pytest.fixture(params=[(1, 2), (1, 2, 3)])
def y_stoich_list(request):
    return request.param


def test_dense_2spec_mixed_order_rxn_stoich(n_species, y_stoich_list):

    reactants, products = chem.create_dense_2_species_mixed_order_reaction_network_stoichiometry(n_species, stoich_list=y_stoich_list)

    n_reactions = n_species * (n_species - 1) * (n_species - 2)
    assert reactants.shape[1] == n_reactions

    matches = chem.find_matching_reactions(reactants, products)

    assert matches.shape[0] == n_reactions / 2

    # Make sure a reactant is never a product
    assert not np.any(np.logical_and(products > 0, reactants > 0))

    # Make sure the stoichiometries are conserved
    assert np.all(np.sum(reactants, axis=0) == np.sum(products, axis=0))

    # Stoihciometry varies but each species should be in the same number of reactions
    assert np.all(np.unique(np.sum(reactants > 0, axis=1)) == np.unique(np.sum(products > 0, axis=1)))


@pytest.fixture(params=[(2, 0), (1, 2)])
def m_n_stoich_pairs(request):
    return request.param


def test_dense_mixed_autocat_rxn_stoich(n_species, m_n_stoich_pairs):

    m, n = m_n_stoich_pairs
    reactants, products = chem.create_dense_mixed_autocatalytic_reaction_network_stoichiometry(n_species, m=m, n=n)

    n_reactions = 2 * n_species * (n_species - 1)
    assert reactants.shape[1] == n_reactions

    # Confirm it's fully reversible
    matches = chem.find_matching_reactions(reactants, products)
    assert matches.shape[0] == n_reactions / 2

    # Make sure the stoichiometries are conserved
    assert np.all(np.sum(reactants, axis=0) == np.sum(products, axis=0))

    # Stoihciometry varies but each species should be in the same number of reactions
    assert np.all(np.unique(np.sum(reactants > 0, axis=1)) == np.unique(np.sum(products > 0, axis=1)))


def test_kitchen_sink_rxn_stoich(n_species):

    reactants, products = chem.create_dense_kitchen_sink_reaction_network_stoichiometry(n_species)

    n_reactions = n_species**2 * (n_species - 1)

    assert reactants.shape[1] == n_reactions

    # Confirm it's fully reversible
    matches = chem.find_matching_reactions(reactants, products)
    assert matches.shape[0] == n_reactions / 2

    # Make sure the stoichiometries are conserved
    assert np.all(np.sum(reactants, axis=0) == np.sum(products, axis=0))

    # Stoihciometry varies but each species should be in the same number of reactions
    assert np.all(np.unique(np.sum(reactants > 0, axis=1)) == np.unique(np.sum(products > 0, axis=1)))


def test_get_complexes(n_species):

    reactants, products = chem.create_dense_reversible_unimolecular_reaction_network_stoichiometry(n_species)
    complexes = chem.get_complexes(reactants, products)

    # This should give one complex for each species
    assert np.all(np.sum(complexes, axis=1) == 1)
    assert complexes.shape[1] == n_species

    n_species = 4
    reactants, products = chem.create_dense_bimolecular_reaction_network_stoichiometry(n_species)
    complexes = chem.get_complexes(reactants, products)

    # We should stoichiometry=1 complexes with each species paired with each other species n_species-1 times
    assert np.all(np.sum(complexes == 1, axis=1) == n_species - 1)
    # And each species in a stoich=2 complex
    assert np.all(np.sum(complexes == 2, axis=1) == 1)
    # Total number of complexes should be n_species*(n_species_1)/2 pairs plus n_species product complexes
    assert complexes.shape[1] == n_species*(n_species - 1) / 2 + n_species


def test_get_complex_names():

    complexes = [[1, 0, 0, 0, 1],
                 [2, 0, 1, 1, 1],
                 [0, 1, 1, 0, 2]]

    names = chem.get_complex_human_names(np.array(complexes))

    assert names == ['A + 2B', 'C', 'B + C', 'B', 'A + B + 2C']


def test_get_complex_edges(n_species):

    reactants, products = chem.create_dense_2_species_mixed_order_reaction_network_stoichiometry(n_species)
    complexes = chem.get_complexes(reactants, products)

    i_react_cpx, i_prod_cpx = chem.get_complex_reaction_edges(reactants, products, complexes)

    # I mean, it's just a convoluted composition of indexing and matching operations so we just reverse it to test..?
    recon_reactants = complexes[:, i_react_cpx]
    recon_prods = complexes[:, i_prod_cpx]
    assert np.all(reactants == recon_reactants)
    assert np.all(products == recon_prods)


def test_get_complex_graph(n_species):

    reactants, products = chem.create_dense_kitchen_sink_reaction_network_stoichiometry(n_species)
    n_reactions = reactants.shape[1]
    complexes = chem.get_complexes(reactants, products)
    complex_names = chem.get_complex_human_names(complexes)
    i_rxt_edges, i_prod_edges = chem.get_complex_reaction_edges(reactants, products, complexes)

    rate_const = np.random.uniform(0.0, 1.0, n_reactions)
    crn = ReactionNetworkTransitionLayer(reactants, products, init_rate_const=rate_const)
    # Call it on some data to initialize it
    crn(np.random.uniform(-.25,.25, size=(3, 32, 32, n_species)))

    graph = chem.get_crn_complex_graph(crn)

    # Check that the number of edges and complexes is right
    assert graph.number_of_edges() == n_reactions
    assert graph.number_of_nodes() == complexes.shape[1]

    # Check the edges
    for i_rxn in range(n_reactions):

        # Check that this reaction exists as a valid edge
        expected_edge = (complex_names[i_rxt_edges[i_rxn]], complex_names[i_prod_edges[i_rxn]])
        edge_attr = graph.edges[expected_edge]

        # Check that the weight corresponds to the rate for the reaction
        assert np.allclose(edge_attr['weight'], crn.rate_const[i_rxn])
        assert np.allclose(edge_attr['weight'], rate_const[i_rxn])
