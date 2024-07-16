import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools as it
import pyvista as pv
from scipy.spatial import KDTree
from math import sqrt
"""
_code_<x>.py: graphs that might be good codes!
x = `archimedes`: The Archimedean solids.

CHOICES:
'truncated_tetrahedron'
'cuboctahedron'
'truncated_cube'
'truncated_octahedron'
'rhombicuboctahedron'
'truncated_cuboctahedron'
'snub_cube'
'icosidodecahedron'
'truncated_dodecahedron'
'truncated_icosahedron'
'rhombicosidodecahedron'
'truncated_icosidodecahedron'
'snub_dodecahedron'
"""

import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Function to create adjacency matrix from edges
def create_adjacency_matrix(edges, num_vertices):
    adjacency_matrix = np.zeros((num_vertices, num_vertices), dtype=int)
    for edge in edges:
        adjacency_matrix[edge[0]][edge[1]] = 1
        adjacency_matrix[edge[1]][edge[0]] = 1
    return adjacency_matrix

# Function to visualize the graph
def visualize_graph(adjacency_matrix):
    G = nx.from_numpy_array(adjacency_matrix)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray')
    plt.savefig("tt.png")

# Example for a truncated tetrahedron (one of the Archimedean solids)
# Vertices: 12, Edges: 18
truncated_tetrahedron_edges = [
    (0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 6), (2, 7), (3, 8), 
    (3, 9), (4, 10), (4, 11), (5, 10), (5, 6), (6, 11), (7, 8), (7, 10), 
    (8, 9), (9, 11)
]

truncated_cube_edges = [(0, 1), (0, 3), (0, 8), (1, 2), (1, 9), (2, 3), (2, 10), (3, 11),
(4, 5), (4, 7), (4, 12), (5, 6), (5, 13), (6, 7), (6, 14), (7, 15),
(8, 9), (8, 16), (9, 10), (9, 17), (10, 11), (10, 18), (11, 19),
(12, 13), (12, 20), (13, 14), (13, 21), (14, 15), (14, 22), (15, 23),
(16, 17), (16, 20), (17, 18), (17, 21), (18, 19), (18, 22), (19, 23),
(20, 21), (21, 22), (22, 23)
]

# Number of vertices for truncated tetrahedron
num_vertices_tt = 12
num_vertices_tc = 24

# Create adjacency matrix
adjacency_matrix_tt = create_adjacency_matrix(truncated_cube_edges, num_vertices_tc)
print(f"Number of vertices = {adjacency_matrix_tt.shape[0]}")
print(f"Number of edges = {np.sum(adjacency_matrix_tt) // 2}")
degrees = set(np.sum(adjacency_matrix_tt, axis=0))
print(f"Degrees = {degrees}")


# Visualize graph
visualize_graph(adjacency_matrix_tt)

# Repeat the above process for other Archimedean solids
# (Define edges for each solid and visualize their adjacency matrices)
