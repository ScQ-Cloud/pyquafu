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
