import numpy as np
"""
_code_<x>.py: graphs that might be good codes!
x = `lattice`: lattices with various periodic boundary conditions

EXPORT FUNCTIONS
`adj_lattice`
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

if __name__ == "__main__":
    m, n = 5, 3
    square_lattice = adj_lattice(m, n, type="square")
    triangular_lattice = adj_lattice(m, n, type="triangular")
    print(f"{(m, n)}-square lattice sum = {round(np.sum(square_lattice) // 2)}")
    print(f"{(m, n)}-triangular lattice sum = {round(np.sum(triangular_lattice) // 2)}")