import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools as it
"""
_code_<x>.py: graphs that might be good codes!
x = `platonic`: the 5 platonic solids as codes, as well as some non-universal covering spaces of their graphs.

EXPORT FUNCTIONS:
`adj_platonic`
`adj_covered_icosahedron`

Tetrahedron:
4 vertices
6 edges
4 faces (triangular)

Cube (Hexahedron):
8 vertices
12 edges
6 faces (square)

Octahedron:
6 vertices
12 edges
8 faces (triangular)

Dodecahedron:
20 vertices
30 edges
12 faces (pentagonal)

Icosahedron:
12 vertices
30 edges
20 faces (triangular)
"""

def adj_platonic(solid_type):
    return nx.adjacency_matrix(platonic_solid_graph(solid_type)).toarray()

def platonic_solid_graph(solid_type):
    if solid_type == "tetrahedron":
        return nx.tetrahedral_graph()
    elif solid_type == "cube":
        return nx.cubical_graph()
    elif solid_type == "octahedron":
        return nx.octahedral_graph()
    elif solid_type == "dodecahedron":
        return nx.dodecahedral_graph()
    elif solid_type == "icosahedron":
        return nx.icosahedral_graph()
    else:
        raise ValueError("Invalid Platonic solid type")

def covered_icosahedron(n):
    """
    Generate the adjacency matrix representation for the graph
    of a n-covering space of the icosahedron.
    """
    assert n >= 2, "n has to be at least 2"
    n_tot = 12 * n
    base_adj = adj_platonic("icosahedron")
    
    # Enumerate the special edges of an icosahedron that will be cut
    special_edges = np.array([
        [11, 5],
        [9, 8],
        [11, 0],
        [7, 8],
        [7,0]
    ])
    is_special = lambda vec: np.any(np.all(special_edges == vec, axis=1))

    # The icosahedron has 12 vertices; make an empty matrix for it
    covering_adj = np.zeros((n_tot, n_tot))

    # For each vertex, connect it to it's same-labeled neighbor
    # unless it's in the special edges, in which case lift 1 level
    # if it's in the 1st column of the special edges, or lower 1 level if 2nd.
    lift = lambda x: (x + 1) % n
    lower = lambda x: (x - 1) % n
    do_nothing = lambda x: x

    for i, j in it.product(range(12), range(12)):
        if base_adj[i, j] == 0:
            continue # don't do any calculations for edges that don't exist in the base graph

        # Raise, lower, or do nothing, depending on the specialty of the edge at hand
        op = do_nothing
        if is_special([i, j]):
            op = lift
        elif is_special([j, i]):
            op = lower

        # Populate the adjacency matrix accordingly
        for label in range(n):
            covering_adj[i*n + label, j*n + op(label)] = 1

    assert np.all(covering_adj == covering_adj.T), "Your adjmat isn't symmetric!"

    return np.array(covering_adj, dtype=np.uint8)

def visualize_platonic_solid(graph):
    plt.figure(figsize=(6, 6))
    nx.draw(graph, with_labels=True, node_color='lightblue', node_size=500,
            font_size=12, font_weight='bold')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Instantiate platonic solids
    tetrahedron = platonic_solid_graph("tetrahedron")
    cube = platonic_solid_graph("cube")
    octahedron = platonic_solid_graph("octahedron")
    dodecahedron = platonic_solid_graph("dodecahedron")
    icosahedron = platonic_solid_graph("icosahedron")

    # Example visualization
    n = 4
    print(f"Icosahedral graph has {np.sum(adj_platonic('icosahedron')) // 2} edges")
    print(f"{n}-covered icosahedral graph has {round(np.sum(covered_icosahedron(n)) // 2)} edges")
    # visualize_platonic_solid(icosahedron)



