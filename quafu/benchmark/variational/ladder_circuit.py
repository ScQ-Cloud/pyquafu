from numpy import random
from quafu.circuits.quantum_circuit import QuantumCircuit

# number of qubits, number of layers
bit_num, n_layers = 4, 2


def ladder_layout_circuit(params, pbc=False):
    """
    `params` is for circuit trainable parameters
    """
    qc = QuantumCircuit(bit_num)
    offset = 0 if pbc else 1
    for j in range(n_layers):
        for i in range(bit_num - offset):
            qc.cnot(i, (i + 1) % bit_num)
        for i in range(bit_num):
            qc.rx(i, params[j, i])
    return qc


def plot():
    para = random.random((n_layers, bit_num))
    qc = ladder_layout_circuit(para)
    qc.plot_circuit(
        title="Ladder Layout for \nVariational Circuit",
        show=True,
        save=False,
    )
