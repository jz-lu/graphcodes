import numpy as np
"""
_code_<x>.py: graphs that might be good codes!
x = `simplex`: a bitstring technique that looks very much like the quantum simplex code

EXPORT FUNCTIONS
`adj_simplex`: 2^k output nodes, k input nodes, connections based on inner product
"""

def adj_simplex(k):
    """
    Construct a 2^k + k graph. 2^k outputs and k inputs.
    Each output is indexed by a length-k string, its binary representation.
    Connect two outputs if their inner product is 0.
    Connect ith input to an output edge if the ith bit of the output is 1.

    We adopt the convention where the inputs are the first k nodes.
    (This is important for input specification in `main.py`.)
    """
    num_nodes = 2**k + k
    adj_mat = np.zeros((num_nodes, num_nodes), dtype=np.uint8)

    # Generate all integers from 0 to 2^k - 1
    numbers = np.arange(2**k, dtype=np.uint8)
    
    # Convert each integer to its binary representation and fill with leading zeros to length k
    bitstrings = ((numbers[:, None] & (1 << np.arange(k))) > 0).astype(np.uint8)

    # Connect output-output edges if their inner product is zero
    for i in range(2**k):
        for j in range(i+1, 2**k):
            if np.mod(bitstrings[i] @ bitstrings[j], 2) == 0:
                adj_mat[i+k, j+k] = 1
                if i != j:
                    adj_mat[i+k, j+k] = 1
                    adj_mat[j+k, i+k] = 1
    
    # Connect input-output edges. Each gets 2^(k-1) edges
    for i in range(k):
        for j in range(2**(k-1)):
            bitstring = format(j, f'0{k-1}b')
            bitstring = bitstring[:i] + '1' + bitstring[i:]
            out_idx = int(bitstring, 2)
            adj_mat[i, k+out_idx] = 1
            adj_mat[k+out_idx, i] = 1

    # Remove the all-zeros output node
    adj_mat = np.delete(adj_mat, k, axis=0)
    adj_mat = np.delete(adj_mat, k, axis=1)
    # print(adj_mat.shape)
    
    assert np.all(adj_mat == adj_mat.T)
    return adj_mat

if __name__ =="__main__":
    print(adj_simplex(3))
