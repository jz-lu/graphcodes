"""
_code_<x>.py: graphs that might be good codes!
x = `ramanujan`: The classical construction of Ramanujan spectral expander graphs via Cayley graphs of PSL(2, Fq)
"""

import numpy as np
import networkx as nx
from itertools import product
import matplotlib.pyplot as plt

# Define the field size q
q = 3  # Example: q is a small prime power

# Generate the finite field elements
F_q = list(range(q))

# Define the special linear group SL(2, F_q)
def sl2_fq():
    SL2_Fq = []
    for a, b, c, d in product(F_q, repeat=4):
        if (a * d - b * c) % q == 1:
            SL2_Fq.append(np.array([[a, b], [c, d]]))
    return SL2_Fq

# Define the projective special linear group PSL(2, F_q)
def psl2_fq():
    SL2_Fq = sl2_fq()
    center = [np.eye(2, dtype=int)]
    PSL2_Fq = []
    for g in SL2_Fq:
        if not any(np.array_equal(g, z*g) for z in center):
            PSL2_Fq.append(g)
    return PSL2_Fq

# Define a set of generators S
def generators_S():
    return [np.array([[1, 1], [0, 1]]), np.array([[1, 0], [1, 1]])]

# Construct the Cayley graph Cay(G, S)
def construct_cayley_graph():
    G = psl2_fq()
    S = generators_S()
    Cayley_G = nx.Graph()
    
    for g in G:
        Cayley_G.add_node(str(g))
    
    for g in G:
        for s in S:
            neighbor = np.dot(g, s) % q
            Cayley_G.add_edge(str(g), str(neighbor))
    
    return Cayley_G

# Generate the Cayley graph
cayley_graph = construct_cayley_graph()

# Compute the adjacency matrix
adj_matrix = nx.adjacency_matrix(cayley_graph).todense()

# Compute eigenvalues of the adjacency matrix
eigenvalues = np.linalg.eigvals(adj_matrix)
print("Eigenvalues:", np.round(np.sort(eigenvalues)[::-1], 2))

# Verify the Ramanujan property
def verify_ramanujan(eigenvalues, d):
    bound = 2 * np.sqrt(d - 1)
    # Exclude the trivial eigenvalue d
    non_trivial_eigenvalues = [ev for ev in eigenvalues if ev != d]
    return all(abs(ev) <= bound for ev in non_trivial_eigenvalues)

# d is the degree of the graph
d = len(generators_S())

# Check if the graph is Ramanujan
is_ramanujan = verify_ramanujan(eigenvalues, d)

# Output results
print(f"Is the graph Ramanujan? {'Yes' if is_ramanujan else 'No'}")

# Visualize the graph
plt.figure(figsize=(10, 10))
nx.draw(cayley_graph, with_labels=True, node_size=700, node_color="lightblue", font_size=10, font_weight="bold")
plt.title("Cayley Graph of PSL(2, F_q)")
plt.show()
