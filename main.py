import networkx as nx
import matplotlib.pyplot as plt
import itertools as it
import numpy as np
import math

from _graph2code import *

from _code_dipyramid import adj_dipyramid
from _code_platonic import *
from _code_simple import adj_simple
from _code_lattice import *
from _code_simplex import adj_simplex
from _code_circles import type_nearest_neighbor
#from _code_archimedes import adj_archimedes
"""
main.py: Main interface for working with graphically generated stabilizer codes.

Basic usage:
(1) Pick from the list of `_code_<x>.py` files a graph you like. 
(2) Familiarize yourself with how to call the `adj_<x>` function and thereby fill the 
    adjacency matrix in the `ADJ_MAT` variable below. 
(3) Decide which nodes you want to be the inputs and put that in the 
    `INPUTS` variable as a list of numbers. 
(4) Run and see what the distance is!
"""

ADJ_MAT = np.load('../cookie_code.npy')
INPUTS = range(6)

d = find_distance_with_hash_table(ADJ_MAT, INPUTS)
print(f"The distance of your chosen graph is at least {d}")
