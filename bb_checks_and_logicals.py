import numpy as np
from _helix_tools import *

def remove_subspace(candidates, stabs):
    n = len(candidates[0])
    nmk = len(row_reduce_mod2(stabs))
    out = np.zeros((0, n), dtype = np.uint8)
    while(len(candidates)) > 0:
        combined = np.append(out, stabs, axis = 0)
        candidate = np.append(combined, [candidates[0]], axis = 0)
        if len(row_reduce_mod2(combined)) != len(row_reduce_mod2(candidate)):
            out = np.append(out, [candidates[0]], axis = 0)
        candidates = candidates[1:]
    return out


diffs = [[0, 1], [0, -1], [1, 0], [-1, 0], [-6, 3], [3, -6]]
zchecks = np.zeros((0, 144), dtype = np.uint8)
xchecks = np.zeros((0, 144), dtype = np.uint8)
for i in range(12):
    for j in range(24):
        if (i + j) % 2 == 1:
            continue # this is not a check
        is_this_a_z_check = (i % 2) == 1
        sign = 1 if is_this_a_z_check else -1
        qubits = np.array([((24 * (i + d[0] * sign) + (j + d[1] * sign) % 24) // 2) % 144 for d in diffs])
        row = np.zeros(144, dtype = np.uint8)
        row[qubits] = 1
        if is_this_a_z_check:
            zchecks = np.append(zchecks, [row], axis = 0)
        else:
            xchecks = np.append(xchecks, [row], axis = 0)

# zchecks = row_reduce_mod2(zchecks)
# xchecks = row_reduce_mod2(xchecks)
np.save("HZ_BB.npy", zchecks)
np.save("HX_BB.npy", xchecks)
zcandidates = null_space_mod2(xchecks)
xcandidates = null_space_mod2(zchecks)
zlogicals = remove_subspace(zcandidates, zchecks)
xlogicals = remove_subspace(xcandidates, xchecks)
