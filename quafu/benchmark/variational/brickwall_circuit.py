# (C) Copyright 2023 Beijing Academy of Quantum Information Sciences
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
"""Brickwall circuit."""

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
