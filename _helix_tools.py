import numpy as np

def symplectic_inner_product_check(matrix):
    n = matrix.shape[1] // 2
    # Vectorized symplectic inner product check
    left = (matrix[:, :n] @ matrix[:, n:].T) % 2
    right = (matrix[:, n:] @ matrix[:, :n].T) % 2
    return np.array_equal(left, right)

def row_reduce_mod2(matrix):
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
    return matrix

def trim_zero_rows(matrix):
    return matrix[~np.all(matrix == 0, axis=1)]

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
    reduced = trim_zero_rows(reduced)

    return reduced

# Example usage:
# input_matrix = np.array([[1, 0, 0, 1], [0, 1, 1, 0]], dtype=np.uint8)
# output_matrix = process_matrix(input_matrix)
# print(output_matrix)
