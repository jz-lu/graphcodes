import numpy as np

def generate_bitstrings(k):
    # Generate all integers from 0 to 2^k - 1
    numbers = np.arange(2**k, dtype=np.uint8)
    
    # Convert each integer to its binary representation and fill with leading zeros to length k
    bitstrings = ((numbers[:, None] & (1 << np.arange(k))) > 0).astype(np.uint8)
    
    return bitstrings

# Example usage
k = 3
bitstrings = generate_bitstrings(k)
print(bitstrings)

print(bitstrings[3] @ bitstrings[1])
