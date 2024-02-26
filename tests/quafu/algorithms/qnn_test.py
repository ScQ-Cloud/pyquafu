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
import pytest
import torch
from quafu.algorithms.ansatz import QuantumNeuralNetwork
from quafu.algorithms.gradients import compute_vjp, jacobian
from quafu.algorithms.interface.torch import TorchTransformer
from quafu.algorithms.templates.basic_entangle import BasicEntangleLayers
from quafu.circuits.quantum_circuit import QuantumCircuit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _generate_random_dataset(num_inputs, num_samples):
    """
    Generate random dataset

    Args:
        num_inputs: dimension of input data
        num_samples: number of samples in the dataset
    """
    # Generate random input coordinates using PyTorch's rand function
    x = 2 * torch.rand([num_samples, num_inputs], dtype=torch.double) - 1

    # Calculate labels based on the sum of input coordinates
    y01 = (torch.sum(x, dim=1) >= 0).to(torch.long)

    # Convert to one-hot vector
    y = torch.zeros(num_samples, 2)  # Two classes (0 and 1)
    y[torch.arange(num_samples), y01] = 1

    # Create a PyTorch dataset
    dataset = TensorDataset(x, y)

    return dataset


class MLP(nn.Module):
    """A simple Multi-Layer Perceptron for test"""

    def __init__(self, num_inputs, num_classes, hidden_size):
        """
        Args:
            num_inputs: input dimension
            num_classes: number of classes
            hidden_size: number of hidden neuron
        """
        super().__init__()

        # Define the layers of the MLP
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, hidden_size, dtype=torch.double),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes, dtype=torch.double),
        )

    def forward(self, x):
        # Propagate the input through the layers
        return self.layers(x)


class ModelStandardCircuit(nn.Module):
    def __init__(self, circ: QuantumCircuit):
        super().__init__()
        self.circ = circ
        num_params = len(circ.parameterized_gates)
        self.linear = nn.Linear(num_params, num_params, dtype=torch.double)

    def forward(self, features):
        out = self.linear(features)
        out = TorchTransformer.execute(self.circ, out, method="external")
        return out


class ModelQuantumNeuralNetwork(nn.Module):
    def __init__(self, circ: QuantumNeuralNetwork):
        super().__init__()
        self.circ = circ

    def forward(self, features):
        out = TorchTransformer.execute(self.circ, features)
        return out


class ModelQuantumNeuralNetworkNative(nn.Module):
    """Test execution of qnn()"""

    def __init__(self, qnn: QuantumNeuralNetwork):
        super().__init__()
        self.qnn = qnn

    def forward(self, features):
        out = self.qnn(features)
        return out

    # def parameters(self, recurse=True):
    #     for p in self.qnn.weights:
    #         yield nn.Parameter(p)


class TestLayers:
    circ = QuantumCircuit(2)
    circ.x(0)
    circ.rx(0, 0.1)
    circ.ry(1, 0.5)
    circ.ry(0, 0.1)

    def _model_grad(self, model, batch_size):
        """Test one forward pass and gradient calculation of a model"""

        # TODO(zhaoyilun): Make out dimension configurable
        features = torch.randn(
            batch_size, 3, requires_grad=True, dtype=torch.double
        )  # batch_size=4, num_params=3
        outputs = model(features)
        targets = torch.randn(batch_size, 2, dtype=torch.double)
        criterion = nn.MSELoss()
        loss = criterion(outputs, targets)
        loss.backward()

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
        criterion = nn.MSELoss()
        loss = criterion(outputs, targets)
        loss.backward()

    def test_torch_layer_qnn(self):
        """Use QuantumNeuralNetwork ansatz"""
        weights = np.random.randn(2, 2)
        entangle_layer = BasicEntangleLayers(weights, 2)
        qnn = QuantumNeuralNetwork(2, [entangle_layer])
        batch_size = 1

        # Legacy invokation style
        model = ModelQuantumNeuralNetwork(qnn)
        self._model_grad(model, batch_size)

        # New invokation style
        model = ModelQuantumNeuralNetworkNative(qnn)
        self._model_grad(model, batch_size)

    @pytest.mark.skip(reason="github env doesn't have token")
    def test_torch_layer_qnn_real_machine(self):
        """Use QuantumNeuralNetwork ansatz"""
        weights = np.random.randn(2, 2)
        entangle_layer = BasicEntangleLayers(weights, 2)
        qnn = QuantumNeuralNetwork(2, [entangle_layer], backend="ScQ-P10")
        qnn.measure([0, 1], [0, 1])
        batch_size = 1

        # New invokation style
        model = ModelQuantumNeuralNetworkNative(qnn)
        self._model_grad(model, batch_size)

    def test_classification_on_random_dataset(self, num_epochs, batch_size):
        """Test e2e hybrid quantum-classical nn training using a synthetic dataset

        Args:
            num_epochs: number of epoches for training
            batch_size: batch size for training

        """
        # Define the hyperparameters
        num_inputs = 2
        num_classes = 2
        learning_rate = 0.01

        # Generate the dataset
        dataset = _generate_random_dataset(num_inputs, 100)

        # Create QNN
        num_qubits = num_classes
        weights = np.random.randn(num_qubits, 2)
        entangle_layer = BasicEntangleLayers(weights, 2)
        qnn = QuantumNeuralNetwork(num_qubits, [entangle_layer])
        # qnn_model = ModelQuantumNeuralNetworkNative(qnn)
        qnn_model = ModelStandardCircuit(qnn)

        # Create MLP
        mlp = MLP(num_inputs, 4, 4)

        # Create hybrid model
        model = nn.Sequential(mlp, qnn_model)
        # model = mlp

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        # Create data loader
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train the model
        for epoch in range(num_epochs):
            for inputs, labels in data_loader:
                # Forward pass
                outputs = model(inputs)

                # Compute the loss
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Update the parameters
                optimizer.step()

            # Print the loss
            print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {loss.item()}")

        # Evaluate the model on the dataset
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.argmax(dim=1)).sum().item()

        print(f"Accuracy: {100 * correct / total:.2f}%")
