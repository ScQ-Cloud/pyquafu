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
import pytest

pytest.importorskip("torch")
import numpy as np  # noqa: E402
from quafu.algorithms import AmplitudeEmbedding, QuantumNeuralNetwork  # noqa: E402
from quafu.algorithms.templates.basic_entangle import BasicEntangleLayers  # noqa: E402


class TestConstructQLayers:
    """Test stacking multiple different quantum layers"""

    def test_construct_qlayers(self):
        state = np.array([7, 2, 3, 4])
        encoding_layer = AmplitudeEmbedding(state=state, num_qubits=2, normalize=True)

        weights = np.array([[-0.850, 1.287], [0.871, 0.184]])
        entangle_layer = BasicEntangleLayers(weights=weights, num_qubits=2)

        circuit = encoding_layer + entangle_layer

        qnn = QuantumNeuralNetwork(2, circuit)

        qnn.draw_circuit(width=2)
