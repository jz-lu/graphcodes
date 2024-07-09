import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from _code_platonic import *

def plot_graph_from_adjacency_matrix(adj_matrix):
    """
    Visualizes a graph given its adjacency matrix using matplotlib and networkx.

    Parameters:
    adj_matrix (numpy.ndarray): A 2D numpy array representing the adjacency matrix of the graph.
    """
    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(np.array(adj_matrix))
    
    # Draw the graph
    pos = nx.spring_layout(G)  # positions for all nodes
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=15)
    plt.show()

# Example usage:
adj_matrix = adj_platonic("icosahedron")
plot_graph_from_adjacency_matrix(adj_matrix)
