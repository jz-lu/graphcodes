import networkx as nx
import matplotlib.pyplot as plt
import itertools as it
import numpy as np
"""
_graph2code.py: find the distance of a code corresponding to a graph

The code is specified by 2 parameters: the graph `adj_mat` in adjacency matrix
form, and a list of indices `inputs` which specify which vertices of the graph are inputs.
From this, we implement a custom F2 linear equation solver to calculate whether a given
physical operator is a stabilizer or not, and whether it commutes with all the stabilizers.
If an operator is not a stabilizer (even up to a sign) BUT commutes with all stabilizers, 
it is in the normalizer of the stabilizer group and therefore a logical operator. 
We enumerate the physical operators in order of lowest to highest weight (arbitrarily
for those of the same weight), and output the distance when we find the first logical operator.

The adjacency matrices of many cool graphs can be found in `_code_<x>.py` files.
"""

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

#def is_stabilizer_signed(g, Stab):
#    """
#    Check if a given Pauli string is a stabilizer of a stabilizer group.
#
#    Input:
#        * g: Pauli string, in symplectic representation (sign | Z's | X's), 2n + 1 bits.
#        * Stab: stabilizer tableau, in symplectic representation (n-k) x (2n + 1) binary matrix.
#
#    Returns:
#        * 0 if g in Stab, 1 if g not in Stab due to Gaussian elimination unsolvability
#        * 2 if -g in Stab but g not in Stab
#    
#    Warnings:
#        * This will not work if the stabilizer tableau has non-independent entries. They
#           must be real generators.
#    """
#    Stab = np.array(Stab, dtype=np.uint8)
#    g = np.array(g, dtype=np.uint8)
#
#    # Drop the signs 
#    g_sgn = g[0]; g_op = g[1:]
#    n = g_op.shape[0] // 2 # number of physical qubits
#    Stab_sgns = Stab[:,0]; Stab_ops = Stab[:,1:]
#    assert len(g.shape) == 1
#    assert g_op.shape[0] == Stab_ops.shape[1]
#    assert g_op.shape[0] % 2 == 0
#    assert Stab_ops.shape[0] < Stab_ops.shape[1] # n-k < n
#
#    # Test for sign-free solvability, using F2 Gaussian elimination + backsubstitution
#    x = solve_F2_eqn(Stab_ops.T, g_op)
#    if x is None:
#        # print("Not GEBS-solvable")
#        return 1 # sign-free unsolvability => unsolvability
#    
#    # Test for signed solvability, by explicitly multiplying the stabilizers together
#    assert np.all(np.mod(np.dot(Stab_ops.T, x), 2) == g_op)
#    running_sgn = 0 # define a running product, which is initialized to the identity, symplectic
#    running_Z = np.zeros(n, dtype=np.uint8)
#    running_X = np.zeros(n, dtype=np.uint8) # running Z and X strings, symplectic
#    for i, bit in enumerate(x):
#        if bit == 1:
#            new_Z_ops = Stab_ops[i,:n]
#            new_X_ops = Stab_ops[i,n:]
#            running_sgn = (running_sgn + Stab_sgns[i] + np.mod(np.dot(new_Z_ops, running_X), 2)) % 2
#            running_Z = np.mod(running_Z + new_Z_ops, 2)
#            running_X = np.mod(running_X + new_X_ops, 2)
#    
#    solution_op = np.hstack((running_Z, running_X), dtype=np.uint8)
#    solution_sgn = running_sgn
#
#    assert np.all(solution_op == g_op), f"Solution {solution_op} != g {g_op}"
#    if g_sgn == solution_sgn:
#        return 0 # correctly solved
#    else:
#        # print("Sign doesn't match")
#        return 2 # sign doesn't match

def is_stabilizer(g, Stab):
    """
    Check if a given Pauli string is a stabilizer of a stabilizer group.

    Input:
        * g: Pauli string, in symplectic representation (Z's | X's), 2n bits.
        * Stab: stabilizer tableau, in symplectic representation (n-k) x 2n binary matrix.

    Returns:
        * True if g in Stab, False if g not in Stab due to Gaussian elimination unsolvability
    
    Warnings:
        * This will not work if the stabilizer tableau has non-independent entries. They
           must be real generators.
        * The generators and inputs are unsigned, so it could be the case that -g is
        in Stab and g is not, yet this will return True.
    """
    Stab = np.array(Stab, dtype=np.uint8)
    g = np.array(g, dtype=np.uint8)

    n = g.shape[0] // 2 # number of physical qubits
    assert len(g.shape) == 1
    assert g.shape[0] == Stab.shape[1]
    assert g.shape[0] % 2 == 0
    assert Stab.shape[0] < Stab.shape[1] # n-k < n

    # Test for sign-free solvability, using F2 Gaussian elimination + backsubstitution
    x = solve_F2_eqn(Stab.T, g)
    if x is None:
        # print("Not GEBS-solvable")
        return False # sign-free unsolvability => unsolvability
    return True

#adj_mat needs to be in KLS form
def find_distance(adj_mat, inputs):
    adj_mat = np.array(adj_mat).tolist()
    pivots = [adj_mat[i].index(1) for i in inputs]
    for i, j in enumerate(inputs):
        for k, l in enumerate(inputs):
            #check if inputs connected
            assert adj_mat[j][l] == 0, f"Inputs {j} and {l} connected"
            #check if pivots connected
            assert adj_mat[pivots[i]][pivots[k]] == 0, f"Pivots of inputs #{i} and #{k} connected"
            #check if pivots next to other inputs
            assert i == k or adj_mat[j][pivots[k]] == 0, f"Input {j} connected to pivot of input #{k}"
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
    symp_stab = np.zeros((n - k, 2 * n), dtype = int)
    for i in range(n - k):
        symp_stab[i][:n] = z_checks[i]
        symp_stab[i][n:] = x_checks[i]
    for cur_dist in range(1, 10):
        print(f"Trying distance {cur_dist}")
        for i in it.combinations(range(n), cur_dist):
            for j in it.product(*[range(1, 4)] * cur_dist):
                error = np.zeros(2 * n, dtype = int)
                for k in range(cur_dist):
                    error[i[k]] = j[k] // 2
                    error[i[k] + n] = j[k] % 2
                error_swap = np.concatenate([error[n:], error[:n]])
                if np.all(np.mod(symp_stab.dot(error_swap), 2) == 0):
                    if not is_stabilizer(error, symp_stab):
                        print(i, j)
                        return cur_dist
    return -1
