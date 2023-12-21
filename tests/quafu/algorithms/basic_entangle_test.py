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
from quafu.algorithms import BasicEntangleLayers
from quafu.circuits import QuantumCircuit


class TestBasicEntangleLayers:
    """Example of building basic_entangle layer"""

    def test_build(self):
        num_qubits = 3
        qc = QuantumCircuit(num_qubits)
        weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        # weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        # weights = np.array([[0.1, 0.2], [0.3, 0.4]])
        qc.add_gates(
            BasicEntangleLayers(weights=weights, num_qubits=num_qubits, rotation="Y")
        )
        qc.draw_circuit(width=num_qubits)

        ##TODO(): if weights are as follows, it need additional processing.
        # weights = np.array([
        #     [
        #         [1, 2, 3, 4],
        #         [5, 6, 7, 8],
        #         [9, 10, 11, 12]
        #     ],
        #     [
        #         [13, 14, 15, 16],
        #         [17, 18, 19, 20],
        #         [21, 22, 23, 24]
        #     ]
        # ])
