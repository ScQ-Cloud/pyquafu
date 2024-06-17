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
from quafu.algorithms import AmplitudeEmbedding
from quafu.circuits import QuantumCircuit


class TestAmplitudeEmbedding:
    """Example of amplitude embedding"""

    def test_build(self):
        num_qubits = 2
        qc = QuantumCircuit(num_qubits)
        state = np.array([6, -12.5, 11.15, 7])
        qc.add_gates(
            AmplitudeEmbedding(state=state, num_qubits=num_qubits, normalize=True)
        )
        qc.draw_circuit(width=num_qubits)

    def test_build_pad(self):
        num_qubits = 2
        qc = QuantumCircuit(num_qubits)
        state = np.array([6, -12.5, 11.15])
        qc.add_gates(
            AmplitudeEmbedding(
                state=state, num_qubits=num_qubits, pad_with=7, normalize=True
            )
        )
        qc.draw_circuit(width=num_qubits)
