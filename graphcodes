import networkx as nx
import matplotlib.pyplot as plt
import itertools as it
import numpy as np

def create_dipyramid(base_polygon):
    """
    Create a dipyramid graph based on the number of sides in the base polygon.
    """
    G = nx.Graph()
    
    # Add base polygon vertices
    for i in range(base_polygon):
        G.add_node(f"B{i}")
    
    # Add top and bottom vertices
    G.add_node("T")
    G.add_node("B")
    
    # Connect base polygon vertices
    for i in range(base_polygon):
        G.add_edge(f"B{i}", f"B{(i+1)%base_polygon}")
        
    # Connect top and bottom vertices to all base vertices
    for i in range(base_polygon):
        G.add_edge("T", f"B{i}")
        G.add_edge("B", f"B{i}")
    
    return G

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

"""
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

def visualize_platonic_solid(graph):
    plt.figure(figsize=(6, 6))
    nx.draw(graph, with_labels=True, node_color='lightblue', node_size=500,
            font_size=12, font_weight='bold')
    plt.axis('off')
    plt.show()

def visualize_dipyramid(graph):
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(graph, k=0.5, iterations=50)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=12, font_weight='bold')
    plt.axis('off')
    plt.show()

# Create dipyramids
triangular_dipyramid = create_dipyramid(3)
square_dipyramid = create_dipyramid(4)
pentagonal_dipyramid = create_dipyramid(5)
hexagonal_dipyramid = create_dipyramid(6)

# # Visualize dipyramids
# visualize_dipyramid(triangular_dipyramid)
# visualize_dipyramid(square_dipyramid)
# visualize_dipyramid(pentagonal_dipyramid)
# visualize_dipyramid(hexagonal_dipyramid)

# Instantiate platonic solids
tetrahedron = platonic_solid_graph("tetrahedron")
cube = platonic_solid_graph("cube")
octahedron = platonic_solid_graph("octahedron")
dodecahedron = platonic_solid_graph("dodecahedron")
icosahedron = platonic_solid_graph("icosahedron")

# # Example visualization
# visualize_platonic_solid(dodecahedron)

# Main idea: return the adjacency matrix
# Here's an example on a dodecahedron
#adj_matrix = nx.adjacency_matrix(dodecahedron).toarray()
#print(adj_matrix)
#visualize_platonic_solid(dodecahedron)

# Here's an example on an octahedral dipyramid
#adj_matrix = nx.adjacency_matrix(create_dipyramid(8)).toarray()
#print(adj_matrix)

def solve_F2_eqn(A, b):
    """
    Solve a matrix equation Ax = b (mod 2), or return None if no solution exists.
    Commented out code solves the case if the (mod 2) is removed.
    """
    # Convert inputs to numpy arrays for efficient operations
    A = np.array(A, dtype=np.uint8)
    b = np.array(b, dtype=np.uint8)
    n = len(b)
    r = A.shape[1]
    assert A.shape[0] == n

    # Combine A and b into an augmented matrix
    augmented = np.column_stack((A, b))

    # ====== BEGIN: Gaussian elimination ======
    # Forward elimination
    for i in range(r):
        # Find pivot
        pivot = np.argmax(np.abs(augmented[i:, i])) + i
        if augmented[pivot, i] == 0:
            print(augmented)
            raise ValueError("Matrix is singular")

        # Swap rows if needed (swap in A as well to track swapping)
        if pivot != i:
            augmented[[i, pivot]] = augmented[[pivot, i]]

        # Eliminate below
        for j in range(i + 1, n):
            # factor = augmented[j, i] / augmented[i, i]
            # augmented[j, i:] -= factor * augmented[i, i:]
            if augmented[j, i] != 0:
                augmented[j, i:] ^= augmented[i, i:] # XOR
    
    # # Normalization
    # for i in range(n):
    #     augmented[i] /= augmented[i,i]
    
    # Check that we did a successful elimination
    for i in range(r):
        assert np.all(augmented[i,:i] == 0) and augmented[i,i] == 1

    # Verify whether there is a solution or not
    for i in range(r, n):
        if np.all(augmented[i, :-1]) == 0 and augmented[i, -1] != 0:
            # print("No solution!")
            # print(augmented)
            return None # No solution

    # ====== END: Gaussian elimination ======

    # ====== BEGIN: Back substitution ======
    x = np.zeros(r, dtype=np.uint8) # the solution vector
    # Go backwards
    for i in range(r - 1, -1, -1):
        # if i == r - 1:
        #     x[i] = augmented[i, -1] / augmented[i, i]
        # else:
        #     x[i] = (augmented[i, -1] - np.dot(augmented[i, i+1:r], x[i+1:])) / augmented[i, i]
        if i == r - 1:
            x[i] = augmented[i, -1]
        else:
            x[i] = augmented[i, -1] ^ np.mod(np.dot(augmented[i, i+1:r], x[i+1:]), 2)
    # ====== END: Back substitution ======

    return x

def is_stabilizer(g, Stab):
    """
    Check if a given Pauli string is a stabilizer of a stabilizer group.

    Input:
        * g: Pauli string, in symplectic representation (sign | Z's | X's), 2n + 1 bits.
        * Stab: stabilizer tableau, in symplectic representation (n-k) x (2n + 1) binary matrix.

    Returns:
        * 0 if g in Stab, 1 if g not in Stab due to Gaussian elimination unsolvability
        * 2 if -g in Stab but g not in Stab
    
    Warnings:
        * This will not work if the stabilizer tableau has non-independent entries. They
           must be real generators.
    """
    Stab = np.array(Stab, dtype=np.uint8)
    g = np.array(g, dtype=np.uint8)

    # Drop the signs 
    g_sgn = g[0]; g_op = g[1:]
    n = g_op.shape[0] // 2 # number of physical qubits
    Stab_sgns = Stab[:,0]; Stab_ops = Stab[:,1:]
    assert len(g.shape) == 1
    assert g_op.shape[0] == Stab_ops.shape[1]
    assert g_op.shape[0] % 2 == 0
    assert Stab_ops.shape[0] < Stab_ops.shape[1] # n-k < n

    # Test for sign-free solvability, using F2 Gaussian elimination + backsubstitution
    x = solve_F2_eqn(Stab_ops.T, g_op)
    if x is None:
        # print("Not GEBS-solvable")
        return 1 # sign-free unsolvability => unsolvability
    
    # Test for signed solvability, by explicitly multiplying the stabilizers together
    assert np.all(np.mod(np.dot(Stab_ops.T, x), 2) == g_op)
    running_sgn = 0 # define a running product, which is initialized to the identity, symplectic
    running_Z = np.zeros(n, dtype=np.uint8)
    running_X = np.zeros(n, dtype=np.uint8) # running Z and X strings, symplectic
    for i, bit in enumerate(x):
        if bit == 1:
            new_Z_ops = Stab_ops[i,:n]
            new_X_ops = Stab_ops[i,n:]
            running_sgn = (running_sgn + Stab_sgns[i] + np.mod(np.dot(new_Z_ops, running_X), 2)) % 2
            running_Z = np.mod(running_Z + new_Z_ops, 2)
            running_X = np.mod(running_X + new_X_ops, 2)
    
    solution_op = np.hstack((running_Z, running_X), dtype=np.uint8)
    solution_sgn = running_sgn

    assert np.all(solution_op == g_op), f"Solution {solution_op} != g {g_op}"
    if g_sgn == solution_sgn:
        return 0 # correctly solved
    else:
        # print("Sign doesn't match")
        return 2 # sign doesn't match

#adj_mat needs to be in KLS form
def find_distance(adj_mat, inputs):
    adj_mat = np.array(adj_mat).tolist()
    pivots = [adj_mat[i].index(1) for i in inputs]
    for i, j in enumerate(inputs):
        for k, l in enumerate(inputs):
            #check if inputs connected
            if adj_mat[j][l] == 1: return -1
            #check if pivots connected
            if adj_mat[pivots[i]][pivots[k]] == 1: return -2
            #check if pivots next to other inputs
            if i != k and adj_mat[j][pivots[k]] == 1: return -3
    outputs = [i for i in range(len(adj_mat)) if i not in inputs]
    io_adj = [[adj_mat[i][j] for j in outputs] for i in inputs]
    oo_adj = [[adj_mat[i][j] for j in outputs] for i in outputs]
    pivots = [i.index(1) for i in io_adj]
    k = len(inputs)
    n = len(outputs)
    z_checks = []
    x_checks = []
    for i in range(n):
        if i not in pivots:
            x_row = np.zeros(n)
            x_row[i] = 1
            z_row = oo_adj[i].copy()
            for j in range(k):
                if io_adj[j][i] == 1:
                    x_row[pivots[j]] = 1
                    z_row = [(z_row[k] + oo_adj[pivots[j]][k]) % 2 for k in range(n)]
            z_checks += [z_row]
            x_checks += [x_row]
    symp_stab = np.zeros((n - k, 2 * n + 1), dtype = int)
    for i in range(n - k):
        for j in range(n):
            symp_stab[i][j + 1] = z_checks[i][j]
            symp_stab[i][n + j + 1] = x_checks[i][j]
        #symp_stab[0] should have sign; currently just has 0
    for cur_dist in range(1, 5):
        for i in it.combinations(range(n), cur_dist):
            for j in it.product(*[range(1, 4)] * cur_dist):
                error = np.zeros(2 * n + 1, dtype = int)
                for k in range(cur_dist):
                    error[i[k] + 1] = j[k] // 2
                    error[n + i[k] + 1] = j[k] % 2
                result = is_stabilizer(error, symp_stab)
                if result == 1:
                    commutes = True
                    for k in symp_stab:
                        parity = True
                        for l in range(n):
                            p1 = k[l + 1], k[n + l + 1]
                            p2 = error[l + 1], error[n + l + 1]
                            if p1 != p2 and p1 != (0, 0) and p2 != (0, 0):
                                parity = not parity
                        if not parity:
                            commutes = False
                    if commutes:
                        print(i, j)
                        return cur_dist
    return -4

fiveqc = [
    [0,1,1,1,1,1],
    [1,0,1,0,0,1],
    [1,1,0,1,0,0],
    [1,0,1,0,1,0],
    [1,0,0,1,0,1],
    [1,1,0,0,1,0]]
sevenqc = [
    [0,0,0,1,0,1,1,0],
    [0,0,0,1,0,1,0,1],
    [0,0,0,1,0,0,1,1],
    [1,1,1,0,0,0,0,0],
    [0,0,0,0,0,1,1,1],
    [1,1,0,0,1,0,0,0],
    [1,0,1,0,1,0,0,0],
    [0,1,1,0,1,0,0,0]]
nineqc = [
    [0,0,0,1,0,0,1,0,0,1],
    [0,0,0,1,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0],
    [1,1,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,1,0,0,0],
    [0,0,0,1,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,1,1,1,0]]

def adj(poly):
    return nx.adjacency_matrix(poly).toarray()

#visualize_platonic_solid(dodecahedron)

mat = create_dipyramid(17)
visualize_dipyramid(mat)
print(adj(mat))
print(find_distance(adj(mat), [0]))

#print(np.array(adj(icosahedron)))
#mat = adj(dodecahedron)
#temp = mat[9]
#mat[9] = mat[12].copy() 
#mat[12] = temp.copy()
#temp[:] = mat[:,9]
#mat[:,9] = mat[:,12].copy() 
#mat[:,12] = temp.copy()
#print(np.array(mat))

#print(find_distance(fiveqc,[0]))
#print(find_distance(sevenqc,[0]))
#print(find_distance(nineqc,[0]))
#print(find_distance(adj(tetrahedron),[0]))
#print(find_distance(adj(cube),[0]))
#print(find_distance(adj(octahedron),[0]))
#print(find_distance(adj(dodecahedron),[0,4,7,12]))
