# Taken from Bravyi's BB code paper
# Commented through by ChatGPT-4o

import numpy as np
from mip import Model, xsum, minimize, BINARY
from bposd.css import css_code  # Module for constructing CSS-type stabilizer codes

# The function `distance_test` computes the minimum Hamming weight of a binary vector x such that:
#   * stab @ x = 0 (mod 2)   --> x is orthogonal to every row of the stabilizer matrix
#   * logicOp @ x = 1 (mod 2) --> x has odd overlap with the logical operator
# Inputs:
#   stab    : numpy array of shape (m, n), binary stabilizer generator matrix
#   logicOp : binary vector of length n (scipy sparse array or numpy), logical operator support
# Output:
#   Returns the distance (minimum weight) of the logical operator in the code

def distance_test(stab, logicOp):
    # Number of physical qubits = number of columns of the stabilizer matrix
    n = stab.shape[1]
    # Number of stabilizer generators = number of rows
    m = stab.shape[0]

    # Maximum weight (number of ones) among all stabilizer rows
    wstab = np.max([np.sum(stab[i, :]) for i in range(m)])
    # Weight of the logical operator (# of nonzero entries). getnnz() works if logicOp is sparse.
    wlog = logicOp.getnnz()

    # To enforce parity constraints (mod 2) via linear constraints, we introduce slack bits.
    # We need enough bits to represent up to wstab or wlog in binary, hence ceil(log2(weight)).
    num_anc_stab = int(np.ceil(np.log2(wstab)))       # ancilla bits per stabilizer row
    num_anc_logical = int(np.ceil(np.log2(wlog)))    # ancilla bits for the logical constraint

    # Total number of binary decision variables:
    #   - n qubit-selection bits x[0..n-1]
    #   - m * num_anc_stab slack bits for stabilizer constraints
    #   - num_anc_logical slack bits for the logical constraint
    num_var = n + m * num_anc_stab + num_anc_logical

    # Create a Mixed-Integer Programming model
    model = Model()
    model.verbose = 0  # suppress solver output

    # Create binary variables x[i] for i=0..num_var-1
    x = [model.add_var(var_type=BINARY) for i in range(num_var)]

    # Objective: minimize the Hamming weight of the qubit vector --> sum x[0..n-1]
    model.objective = minimize(xsum(x[i] for i in range(n)))

    # --- Constraints for orthogonality to each stabilizer row (mod 2) ---
    for row in range(m):
        # Build coefficient vector for the linear constraint
        weight = [0] * num_var
        # Support of the row: indices where stab[row, :] == 1
        supp = np.nonzero(stab[row, :])[0]
        # For each qubit bit in that support, add +1 * x[q]
        for q in supp:
            weight[q] = 1

        # Now encode parity-mod-2 using slack bits:
        # We want sum_{q in supp} x[q] ≡ 0 (mod 2).
        # Introduce ancilla binary variables representing the binary expansion of the sum.
        # We enforce: sum(supp bits) - sum(2^k * ancilla_k) == 0.
        cnt = 1
        for k in range(num_anc_stab):
            idx = n + row * num_anc_stab + k
            weight[idx] = -(1 << cnt)  # subtract 2^cnt * x[idx]
            cnt += 1

        # Add equality constraint
        model += xsum(weight[i] * x[i] for i in range(num_var)) == 0

    # --- Constraint for odd overlap with the logical operator ---
    # Support of logicOp: qubits touched by logical operator
    supp = np.nonzero(logicOp)[0]
    print("Logical Supp =", supp)
    print("Logical op =", logicOp)
    weight = [0] * num_var
    for q in supp:
        weight[q] = 1

    # Similar binary encoding: enforce sum(supp bits) ≡ 1 (mod 2)
    cnt = 1
    base_idx = n + m * num_anc_stab
    for k in range(num_anc_logical):
        weight[base_idx + k] = -(1 << cnt)
        cnt += 1

    # Add equality constraint: sum == 1
    model += (xsum(weight[i] * x[i] for i in range(num_var)) == 1)

    # Solve the MIP
    model.optimize()

    # Extract the optimal weight: sum of qubit bits x[0..n-1]
    opt_val = sum(int(x[i].x) for i in range(n))
    sol = np.array([int(x[i].x) for i in range(n)], dtype=int)
    print("Weight =", weight)
    print("Constraint value =", sum(int(x[i].x) * weight[i] for i in range(num_var)))
    print("Logical overlap =", sum(int(x[i].x) * weight[i] for i in range(n)))
    return int(opt_val), sol

# Note:
# - `mip` is the Python-MIP package for mixed-integer programming (Model, xsum, minimize, BINARY).
# - CSS codes (Calderbank-Shor-Steane) are accessed via `bposd.css.css_code`, but not shown in this snippet.

# [[144,12,12]]
ell,m = 12,6
a1,a2,a3 = 3,1,2
b1,b2,b3 = 3,1,2


n = 2*ell*m
n2 = ell*m


# define cyclic shift matrices 
I_ell = np.identity(ell,dtype=int)
I_m = np.identity(m,dtype=int)
I = np.identity(ell*m,dtype=int)
x = {}
y = {}
for i in range(ell):
	x[i] = np.kron(np.roll(I_ell,i,axis=1),I_m)
for i in range(m):
	y[i] = np.kron(I_ell,np.roll(I_m,i,axis=1))

# define check matrices
A = (x[a1] + y[a2] + y[a3]) % 2
B = (y[b1] + x[b2] + x[b3]) % 2
AT = np.transpose(A)
BT = np.transpose(B)
hx = np.hstack((A,B))
hz = np.hstack((BT,AT))

HX_BB = np.load("HX_BB.npy")
HZ_BB = np.load("HZ_BB.npy")
hx = HX_BB
hz = HZ_BB

# qcode=css_code(hx,hz)
qcode = css_code(hx, hz)
print('Testing CSS code...')
qcode.test()
print('Done')

lz = qcode.lz
lx = qcode.lx
k = lz.shape[0]

print('Computing code distance...')
# We compute the distance only for Z-type logical operators (the distance for X-type logical operators is the same)
# by solving an integer linear program (ILP). The ILP looks for a minimum weight Pauli Z-type operator which has an even overlap with each X-check 
# and an odd overlap with logical-X operator on the i-th logical qubit. Let w_i be the optimal value of this ILP. 
# Then the code distance for Z-type logical operators is dZ = min(w_1,…,w_k).
d = n
k = 1
for i in range(k):
	# w = distance_test(hx,lx[i,:])
	# print('Logical qubit=',i,'Distance=',w)
	w, sol = distance_test(hx, lx[i,:])
	print("  found weight‐", w, "with support", np.nonzero(sol)[0])

	# now check consistency by brute force:
	print("  hx @ sol mod2 =", (hx.dot(sol) % 2))
	print("  lx[i] @ sol mod2 =", (lx[i,:].dot(sol) % 2))
	d = min(d,w)

print('Code parameters: n,k,d=',n,k,d)