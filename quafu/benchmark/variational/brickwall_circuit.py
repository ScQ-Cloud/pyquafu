from numpy import random
from quafu.circuits.quantum_circuit import QuantumCircuit

qubit_num = 6
n_layers = 2


def brickwall_layout_circuit(params, pbc=False):
    """
    `params` is for circuit trainable parameters
    """
    c = QuantumCircuit(qubit_num)
    offset = 0 if pbc else 1
    for j in range(n_layers):
        for i in range(0, qubit_num - offset, 2):
            c.cnot(i, (i + 1) % qubit_num)
        for i in range(qubit_num):
            c.rx(i, params[j, i, 0])
        for i in range(1, qubit_num - offset, 2):
            c.cnot(i, (i + 1) % qubit_num)
        for i in range(qubit_num):
            c.rx(i, params[j, i, 1])
    return c


def plot():
    para = random.random((n_layers, qubit_num, 2))
    qc = brickwall_layout_circuit(para)
    qc.plot_circuit(
        title="Brickwall Layout for \nVariational Circuit",
        show=True,
        save=False,
    )


plot()
