import time

import stim


def make_code() -> tuple[list[stim.PauliString], list[stim.PauliString], list[stim.PauliString]]:

    stabilizers = [stim.PauliString(x) for x in ['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ']]

    completed_tableau = stim.Tableau.from_stabilizers(
        stabilizers,
        allow_redundant=True,
        allow_underconstrained=True,
    )
    obs_indices = [
        k
        for k in range(len(completed_tableau))
        if completed_tableau.z_output(k) not in stabilizers
    ]
    observable_xs: list[stim.PauliString] = [
        completed_tableau.x_output(k)
        for k in obs_indices
    ]
    observable_zs: list[stim.PauliString] = [
        completed_tableau.z_output(k)
        for k in obs_indices
    ]

    return stabilizers, observable_xs, observable_zs


def make_circuit(
    stabilizers: list[stim.PauliString],
    obs_zs: list[stim.PauliString],
    obs_xs: list[stim.PauliString]
) -> stim.Circuit:
    num_qubits = len(stabilizers[0])
    circuit = stim.Circuit()

    for k, observable in enumerate(obs_zs):
        circuit.append("MPP", stim.target_combined_paulis(observable))

    for stabilizer in stabilizers:
        circuit.append('MPP', stim.target_combined_paulis(stabilizer))

    circuit.append('DEPOLARIZE1', range(num_qubits), 1e-3)

    for k, stabilizer in enumerate(stabilizers):
        circuit.append('MPP', stim.target_combined_paulis(stabilizer))
        circuit.append('DETECTOR', [stim.target_rec(-5), stim.target_rec(-1)])

    for k, observable in enumerate(obs_zs):
        circuit.append("MPP", stim.target_combined_paulis(observable))
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-10), stim.target_rec(-1)], k)

    return circuit


def main():
    t0 = time.monotonic()

    stabilizers, obs_xs, obs_zs = make_code()
    circuit = make_circuit(stabilizers, obs_zs, obs_xs)
    wcnf_string = circuit.shortest_error_sat_problem(format='WDIMACS')
    t1 = time.monotonic()
    print(f"Problem created in {t1 - t0:0.3f}s")

    from pysat.examples.rc2 import RC2
    from pysat.formula import WCNF
    wcnf = WCNF(from_string=wcnf_string)
    with RC2(wcnf) as rc2:
        rc2.compute()
        print(f"distance = {rc2.cost}")
    t2 = time.monotonic()
    print(f"Problem solved in {t2 - t1:0.3f}s")


if __name__ == '__main__':
    main()
