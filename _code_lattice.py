import numpy as np
"""
_code_<x>.py: graphs that might be good codes!
x = `lattice`: lattices with various periodic boundary conditions

EXPORT FUNCTIONS
`adj_lattice`: 2D triangular and square lattices, with toric boundary conditions
`adj_3lattice`: k copies of a 3D cubic grid glued together, with toric boudnary conditions
"""

def adj_lattice(m, n, type='triangular'):
    """
    Create a triangular or square lattice with `n` nodes per row, and `m` rows,
    with toric periodic boundary conditions.
    """
    assert type in ["triangular", "square"], f"{type} is an invalid type"
    N = m*n
    adj_mat = np.zeros((N, N))

    def pair2idx(pair):
        """
        Convert a coordinate `pair` = `(a, b)` into an aggregate index for the adj mat.
        a runs through [m] and b runs through [n]
        """
        a, b = pair
        return n*a + b

    def right(a, b):
        return (a, (b + 1) % n)
    def up(a, b):
        return ((a + 1) % m, b)
    def upright(a, b):
        return ((a + 1) % m, (b + 1) % n)
    
    # Fill in a canonically directed manner first (up, right, up-right if triangular)
    # then once done just add the transpose to make it undirected.
    # Note that I'm working in a weird coordinate system where `right` is 
    # spatially right, but `up` is spatially down. This does not matter in the end.
    for i in range(m):
        for j in range(n):
            nbrs = [right(i, j), up(i, j)]
            if type == "triangular":
                nbrs.append(upright(i, j))
            pair = (i, j)
            for nbr in nbrs:
                adj_mat[pair2idx(pair), pair2idx(nbr)] = 1
    
    assert np.all(np.diag(adj_mat) == 0), f"{np.diag(adj_mat)}"
    adj_mat = adj_mat + adj_mat.T
    return np.array(adj_mat, dtype=np.uint8)

def cubic_lattice(n1, n2, n3):
    """
    Make a cubic lattice in 3D with n1, n2, n3 nodes on each dimension, with toric boundary conditions.
    """
    n_tot = n1 * n2 * n3
    adjmat = np.zeros((n_tot, n_tot), dtype=np.uint8)

    def triple2idx(triple):
        a, b, c = triple
        return a*(n2*n3) + b*n3 + c
    
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                triple = (i, j, k)
                nb1 = ((i + 1) % n1, j, k)
                nb2 = ((i - 1) % n1, j, k)
                nb3 = (i, (j + 1) % n2, k)
                nb4 = (i, (j - 1) % n2, k)
                nb5 = (i, j, (k + 1) % n3)
                nb6 = (i, j, (k - 1) % n3)
                nbs = [nb1, nb2, nb3, nb4, nb5, nb6]
                for nb in nbs:
                    adjmat[triple2idx(triple), triple2idx(nb)] = 1

    assert np.all(np.diag(adjmat) == 0)
    return adjmat
                
def adj_3lattice(n1, n2, n3):
    """
    Start with a 3D cubic lattice. Make 2 copies. Glue them together
    by adding a local edge between each copy of the same node. Add toric boundary 
    conditions.

    Degree = 7
    """
    n_tot = n1*n2*n3
    adjmat = np.zeros((2*n_tot, 2*n_tot), dtype=np.uint8)
    cubic = cubic_lattice(n1, n2, n3)
    adjmat[:n_tot,:n_tot] = cubic
    adjmat[n_tot:,n_tot:] = cubic

    def triple2idx(triple):
        a, b, c = triple
        return a*(n2*n3) + b*n3 + c
    
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                    idx = triple2idx((i, j, k))
                    adjmat[idx, idx + n_tot] = 1
                    adjmat[idx + n_tot, idx] = 1
    
    assert np.all(np.diag(adjmat) == 0)
    assert np.all(adjmat == adjmat.T)
    return adjmat


if __name__ == "__main__":
    m, n = 5, 3
    square_lattice = adj_lattice(m, n, type="square")
    triangular_lattice = adj_lattice(m, n, type="triangular")
    print(f"{(m, n)}-square lattice sum = {round(np.sum(square_lattice) // 2)}")
    print(f"{(m, n)}-triangular lattice sum = {round(np.sum(triangular_lattice) // 2)}")

    n1, n2, n3 = 3, 3, 4
    glued_3cube = adj_3lattice(n1, n2, n3)
    print(f"{(n1, n2 , 3)}-glued 3-cube lattice sum = {round(np.sum(glued_3cube) // 2)}")