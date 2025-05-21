import numpy as np
from itertools import combinations
from sympy.combinatorics.named_groups import *

def group_inverses(group):
    return [g.inverse() for g in group.elements]

def group_dict(group):
    indices = {}
    for i, g in enumerate(group.elements):
        indices[g] = i
    return indices

def group_properties(group):
    return group.elements, group_inverses(group), group_dict(group)

def group_stabilizer(elements, inverses, indices, zs, xs):
    n = len(elements)
    stabilizers = np.zeros((n, 2 * n), dtype = np.uint8)
    for i range(n):
        for z in elements[z]:
            stabs[i, indices[z * elements[i]]] = 1
        for x in elements[x]:
            stabs[i, n + indices[x * inverses[i]]] = 1
    return stabilizers

def try_all(group, max_total_weight = 8, max_z_weight = 7, max_x_weight = 7):
    elem, inv, idx = group_properties(group)
    n = len(elem)
    for z_weight in range(1, min(max_total_weight - 1, max_z_weight)):
        z_sets = list(combinations(range(n), z_weight))
        for x_weight in range(1, min(max_x_weight, max_total_weight - z_weight)):
            x_sets = list(combinations(range(n), x_weight))
            for zs in z_sets:
                for xs in x_sets:
                    group_stabilizer(elem, inv, idx, zs, xs)
