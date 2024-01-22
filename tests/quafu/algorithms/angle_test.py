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

import numpy as np
import quafu.elements.element_gates as qeg
from quafu.algorithms import AngleEmbedding
from quafu.circuits import QuantumCircuit


class TestAngleEmbedding:
    """Example of angle embedding"""

    def test_build(self):
        num_qubits = 4
        qc = QuantumCircuit(num_qubits)
        feature = np.array([[6, -12.5, 11.15, 7], [8, 9.5, -11, -5], [5, 0.5, 8, -7]])
        for i in range(4):
            qc.add_ins(qeg.HGate(pos=i))
        for i in range(len(feature)):
            qc.add_gates(
                AngleEmbedding(features=feature[i], num_qubits=num_qubits, rotation="Y")
            )
        qc.draw_circuit(width=num_qubits)
