from scipy.stats import unitary_group
from quafu import QuantumCircuit

nqubit = 5
qubits = list(range(nqubit))
U0 = unitary_group.rvs(2 ** nqubit)

# Using QSD to decompose the unitary
qc = QuantumCircuit(nqubit)
qc.unitary(U0, qubits)
qc.draw_circuit()
