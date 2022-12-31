"""
Under construction....
"""
import os

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import chem
import utils
from analysis import get_compound_model_components
from models.transition import dense_gs_crn


# n_species = 5
# x_shape = (64, 64, n_species)
#crn = dense_gs_crn(n_species)
#crn.build((9,) + x_shape)

#model_dir = '/Users/hunterelliott/Iteration_C1/Dense_CRNs/v2_Max_Diss/test_1/Nt32'
model_dir = '/Users/hunterelliott/Iteration_C1/Dense_CRNs/v4_Replication/test_84'

model = utils.load_model(model_dir)

crn = get_compound_model_components(model)[1]



# Convert our CRN into a networkX graph
complexes = chem.get_complexes(crn.reactants, crn.products)
complex_names = chem.get_complex_human_names(complexes)
i_rxt_edges, i_prod_edges = chem.get_complex_reaction_edges(crn.reactants, crn.products, complexes)
named_edges = [(complex_names[i_rxt], complex_names[i_prod]) for (i_rxt, i_prod) in zip(i_rxt_edges, i_prod_edges)]
norm_weights = []
max_rate = np.max(crn.rate_const)
min_rate = np.min(crn.rate_const)
for i_edge in range(len(named_edges)):
    #norm_weights.append(np.sum(crn.rate_const <=  crn.rate_const[i_edge]) / crn.rate_const.shape[0])
    norm_weights.append(float((crn.rate_const[i_edge] - min_rate) / (max_rate - min_rate)))

edges_and_weights = [edge + ({'weight':weight},) for (weight, edge) in zip(norm_weights, named_edges)]
G = nx.DiGraph()
G.add_nodes_from(complex_names)

G.add_edges_from(edges_and_weights)

fig = plt.figure(figsize=(10,6))


#pos = nx.spring_layout(G, k=1/complexes.shape[1]**(1/2) * 600, iterations=100000)
pos = nx.spring_layout(G, k=1/complexes.shape[1]**(1/2) * 5, iterations=100)
#pos = nx.circular_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=900, alpha=.6)
nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
edges = nx.draw_networkx_edges(G, pos, edgelist=named_edges, width=2, arrowstyle='->', arrowsize=20, connectionstyle='arc3,rad=0.2')
min_edge_width=2
max_edge_width=6
min_alpha = .05
for i_edge in range(G.number_of_edges()):
    edges[i_edge].set_alpha(max(min_alpha,norm_weights[i_edge]))
    edges[i_edge].set_linewidth(max(norm_weights[i_edge]*max_edge_width, min_edge_width))

plt.show()

fig.savefig(os.path.join(model_dir, "reaction_graph.png"), dpi=300)


