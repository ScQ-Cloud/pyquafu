# (C) Copyright 2024 Beijing Academy of Quantum Information Sciences
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import numpy as np
from quafu.circuits.quantum_circuit import QuantumCircuit


def get_const_oracle(qc: QuantumCircuit):
    n = qc.num - 1
    output = np.random.randint(2)
    if output == 1:
        qc.x(n)
    qc.name = "Constant Oracle"
    return qc


def get_balanced_oracle(qc: QuantumCircuit):
    n = qc.num - 1
    b_str = "".join([random.choice("01") for _ in range(n)])

    # Place X-qu_gate
    for qubit, s in enumerate(b_str):
        if s == "1":
            qc.x(qubit)

    # Use barrier as divider
    qc.barrier()

    # Controlled-NOT qu_gate
    for qubit in range(n):
        qc.cnot(qubit, n)

    qc.barrier()

    # Place X-qu_gate
    for qubit, s in enumerate(b_str):
        if s == "1":
            qc.x(qubit)

    qc.name = "Balanced Oracle"
    return qc


def deutsch_jozsa(n: int, case: str):
    circuit = QuantumCircuit(n + 1)  # number of q-bit and c-bit
    # Initialization
    for qubit in range(n):
        circuit.h(qubit)
    circuit.x(n)
    circuit.h(n)

    # Add oracle
    #################################################
    if case == "balanced":
        get_balanced_oracle(circuit)
    elif case == "constant":
        get_const_oracle(circuit)
    else:
        raise ValueError("undefined case: " + case)
    #################################################

    # Repeat H-qu_gate
    circuit.barrier()
    for qubit in range(n):
        circuit.h(qubit)
    circuit.barrier()

    # Measure
    circuit.measure(list(range(n)), list(range(n)))
    return circuit


dj_qc = deutsch_jozsa(n=4, case="constant")
dj_qc.plot_circuit(title="Deutsch-Josza Circuit", show=True)
