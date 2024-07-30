import networkx as nx
import matplotlib.pyplot as plt
import itertools as it
import numpy as np

from _graph2code import find_distance

from _code_dipyramid import adj_dipyramid
from _code_platonic import *
from _code_simple import adj_simple
from _code_lattice import *
from _code_simplex import adj_simplex
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

#k = 6
#ADJ_MAT = adj_simplex(k)
#INPUTS = list(range(k))

#PARAMS [16, 4, 3]
#ADJ_MAT=adj_platonic('dodecahedron')
#visualize_platonic_solid(platonic_solid_graph('dodecahedron'))
#INPUTS = [0, 6, 13, 17]

#PARAMS [22, 2, 5]
#ADJ_MAT = adj_covered_icosahedron(2)
#visualize_platonic_solid(platonic_solid_graph('icosahedron'))
#INPUTS = [0, 0 + 12]

#PARAMS [54, 6, 5]
#ADJ_MAT = adj_covered_icosahedron(5)
#INPUTS = [10, 6 + 12, 8 + 24, 2 + 24, 11 + 36, 5 + 48]

#PARAMS [48, 1, 6]
#ADJ_MAT = adj_lattice(7, 7)
#INPUTS = [0]

#PARAMS [44, 1, 6]
#ADJ_MAT = cubic_lattice(5, 3, 3)
#INPUTS = [0]

#POSSIBLE PARAMS [53, 1, 7]
ADJ_MAT = adj_3lattice(3, 3, 3)
INPUTS = [0]

d = find_distance(ADJ_MAT, INPUTS)
print(f"The distance of your chosen graph is {d}")
