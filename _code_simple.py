import numpy as np
"""
_code_<x>.py: graphs that might be good codes!
x = `simple`: the 5-qubit, 7-qubit, and 9-qubit codes we all know and love.
"""

fiveqc = np.array([
    [0,1,1,1,1,1],
    [1,0,1,0,0,1],
    [1,1,0,1,0,0],
    [1,0,1,0,1,0],
    [1,0,0,1,0,1],
    [1,1,0,0,1,0]], dtype=np.uint8)

sevenqc = np.array([
    [0,0,0,1,0,1,1,0],
    [0,0,0,1,0,1,0,1],
    [0,0,0,1,0,0,1,1],
    [1,1,1,0,0,0,0,0],
    [0,0,0,0,0,1,1,1],
    [1,1,0,0,1,0,0,0],
    [1,0,1,0,1,0,0,0],
    [0,1,1,0,1,0,0,0]], dtype=np.uint8)

nineqc = np.array([
    [0,0,0,1,0,0,1,0,0,1],
    [0,0,0,1,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0],
    [1,1,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,1,0,0,0],
    [0,0,0,1,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,1,1,1,0]], dtype=np.uint8)

def adj_simple(n):
    assert n in [5, 7, 9], f"{n} is not a valid code number, try 5, 7, or 9"
    if n == 5:
        return fiveqc
    elif n == 7:
        return sevenqc
    else:
        return nineqc
