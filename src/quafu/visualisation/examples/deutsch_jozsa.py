import random

import matplotlib.pyplot as plt
import numpy as np

from quafu.circuits.quantum_circuit import QuantumCircuit
from ..circuitPlot import CircuitPlotManager


def get_const_oracle(n: int):
    const_oracle = QuantumCircuit(n + 1)
    output = np.random.randint(2)
    if output == 1:
        const_oracle.x(n)
    const_oracle.name = 'Constant Oracle'
    return const_oracle


def get_balanced_oracle(n: int):
    oracle = QuantumCircuit(n + 1)
    b_str = ''.join([random.choice('01') for _ in range(n)])

    # Place X-qu_gate
    for qubit in range(len(b_str)):
        if b_str[qubit] == '1':
            oracle.x(qubit)

    # Use barrier as divider
    oracle.barrier(list(range(n+1)))

    # Controlled-NOT qu_gate
    for qubit in range(n):
        oracle.cnot(qubit, n)

    oracle.barrier(list(range(n+1)))

    # Place X-qu_gate
    for qubit in range(len(b_str)):
        if b_str[qubit] == '1':
            oracle.x(qubit)

    oracle.name = 'Balanced Oracle'
    return oracle


def deutsch_jozsa(n: int, case: str):
    circuit = QuantumCircuit(n + 1)  # number of q-bit and c-bit

    # Initialization
    for qubit in range(n):
        circuit.h(qubit)
    circuit.x(n)
    circuit.h(n)

    # Add oracle
    #################################################
    if case == 'balanced':
        b_str = ''.join([random.choice('01') for _ in range(n)])

        # Place X-qu_gate
        for qubit in range(len(b_str)):
            if b_str[qubit] == '1':
                circuit.x(qubit)

        circuit.barrier(list(range(n+1)))
        # Controlled-NOT qu_gate
        for qubit in range(n):
            circuit.cnot(qubit, n)
        circuit.barrier(list(range(n+1)))

        # Place X-qu_gate
        for qubit in range(len(b_str)):
            if b_str[qubit] == '1':
                circuit.x(qubit)
    elif case == 'constant':
        const_oracle = QuantumCircuit(n + 1)
        output = np.random.randint(2)
        if output == 1:
            const_oracle.x(n)
    else:
        raise ValueError('undefined case: ' + case)
    #################################################

    # Repeat H-qu_gate
    circuit.barrier(list(range(n+1)))
    for qubit in range(n):
        circuit.h(qubit)
    circuit.barrier(list(range(n+1)))

    # Measure
    circuit.measure(list(range(n)), list(range(n)))
    return circuit


if __name__ == '__main__':
    # plt.figure(dpi=240)
    # balanced_oracle = get_balanced_oracle(n=4)
    circuits_ = deutsch_jozsa(n=4, case='constant')
    cmp = CircuitPlotManager(circuits_)
    cmp(title='Deutsch-Josza Circuit')
    import os

    if not os.path.exists('./figures/'):
        os.mkdir('../figures/')
    plt.savefig('../figures/deutsch_jozsa.png', dpi=240)
    # plt.show()
