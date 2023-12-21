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
import torch.nn
from quafu.algorithms import QuantumNeuralNetwork
from quafu.algorithms.gradients import compute_vjp, jacobian
from quafu.algorithms.interface.torch import execute
from quafu.algorithms.templates.basic_entangle import BasicEntangleLayers
from quafu.circuits.quantum_circuit import QuantumCircuit


class ModelStandardCircuit(torch.nn.Module):
    def __init__(self, circ: QuantumCircuit):
        super().__init__()
        self.circ = circ
        self.linear = torch.nn.Linear(3, 3, dtype=torch.double)

    def forward(self, features):
        out = self.linear(features)
        out = execute(self.circ, out, method="external")
        return out


class ModelQuantumNeuralNetwork(torch.nn.Module):
    def __init__(self, circ: QuantumNeuralNetwork):
        super().__init__()
        self.circ = circ

    def forward(self, features):
        out = execute(self.circ, features)
        return out


class TestLayers:
    circ = QuantumCircuit(2)
    circ.x(0)
    circ.rx(0, 0.1)
    circ.ry(1, 0.5)
    circ.ry(0, 0.1)

    def test_compute_vjp(self):
        params_input = np.random.randn(4, 3)
        jac = jacobian(self.circ, params_input)

        dy = np.random.randn(4, 2)
        vjp = compute_vjp(jac, dy)

        assert len(vjp.shape) == 2
        assert vjp.shape[0] == 4

    def test_torch_layer_standard_circuit(self):
        batch_size = 1
        model = ModelStandardCircuit(self.circ)
        features = torch.randn(
            batch_size, 3, requires_grad=True, dtype=torch.double
        )  # batch_size=4, num_params=3
        outputs = model(features)
        targets = torch.randn(batch_size, 2, dtype=torch.double)
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, targets)
        loss.backward()

    def test_torch_layer_qnn(self):
        """Use QuantumNeuralNetwork ansatz"""
        weights = np.random.randn(2, 2)
        entangle_layer = BasicEntangleLayers(weights, 2)
        qnn = QuantumNeuralNetwork(2, [entangle_layer])
        batch_size = 1
        model = ModelQuantumNeuralNetwork(qnn)
        features = torch.randn(
            batch_size, 3, requires_grad=True, dtype=torch.double
        )  # batch_size=4, num_params=3
        outputs = model(features)
        targets = torch.randn(batch_size, 2, dtype=torch.double)
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, targets)
        loss.backward()
