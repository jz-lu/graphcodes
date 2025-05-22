import numpy as np

def symplectic_inner_product_check(matrix):
    n = matrix.shape[1] // 2
    # Vectorized symplectic inner product check
    left = (matrix[:, :n] @ matrix[:, n:].T) % 2
    right = (matrix[:, n:] @ matrix[:, :n].T) % 2
    return np.array_equal(left, right)

def row_reduce_mod2(matrix, trim_zero_rows = True):
    matrix = matrix.copy()
    m, n = matrix.shape
    row = 0
    for col in range(n):
        if row >= m:
            break
        pivot = None
        for r in range(row, m):
            if matrix[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            matrix[[row, pivot]] = matrix[[pivot, row]]
        for r in range(m):
            if r != row and matrix[r, col]:
                matrix[r] ^= matrix[row]
        row += 1
    if trim_zero_rows:
        return matrix[~np.all(matrix == 0, axis=1)]
    return matrix

def process_matrix(matrix, skip_checks=False):
    if not skip_checks:
        assert isinstance(matrix, np.ndarray), "Input must be a numpy array"
        assert matrix.ndim == 2, "Matrix must be 2D"
        assert np.all((matrix == 0) | (matrix == 1)), "Matrix must contain only 0s and 1s"
        rows, cols = matrix.shape
        assert cols % 2 == 0, "Matrix must have an even number of columns"
        assert symplectic_inner_product_check(matrix), "Matrix does not satisfy symplectic inner product condition"

    # Row reduction over F2
    reduced = row_reduce_mod2(matrix)

    return reduced

def null_space_mod2(A):
    """
    Given a binary numpy array A, returns a matrix whose rows form a basis 
    for the subspace orthogonal to the row space of A over F_2.
    Assumes availability of a function `row_reduce_mod2` that performs row 
    reduction over F_2 and removes zero rows.
    """
    m, n = A.shape
    A_rref = row_reduce_mod2(A)

    # Track pivot columns
    pivots = []
    row = 0
    for col in range(n):
        if row < A_rref.shape[0] and A_rref[row, col] == 1:
            pivots.append(col)
            row += 1

    free_vars = [j for j in range(n) if j not in pivots]
    basis = []

    for free in free_vars:
        vec = np.zeros(n, dtype=np.uint8)
        vec[free] = 1
        # Back substitute to satisfy equations in reduced rows
        for i in reversed(range(len(pivots))):
            pivot_col = pivots[i]
            val = A_rref[i, free] if free < A_rref.shape[1] else 0
            vec[pivot_col] = val % 2
        basis.append(vec)

    return np.array(basis, dtype=np.uint8)
