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

pytest.importorskip("torch")
import torch  # noqa: E402
from quafu.algorithms.ansatz import QuantumNeuralNetwork  # noqa: E402
from quafu.algorithms.gradients import compute_vjp, jacobian  # noqa: E402
from quafu.algorithms.interface.torch import (  # noqa: E402
    ModuleWrapper,
    TorchTransformer,
)
from quafu.algorithms.templates.amplitude import AmplitudeEmbedding  # noqa: E402
from quafu.algorithms.templates.angle import AngleEmbedding  # noqa: E402
from quafu.algorithms.templates.basic_entangle import BasicEntangleLayers  # noqa: E402
from quafu.circuits.quantum_circuit import QuantumCircuit  # noqa: E402
from quafu.elements import Parameter  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402


def _generate_random_dataset(num_inputs, num_samples, one_hot: bool = True):
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
    if one_hot:
        return TensorDataset(x, y)
    return TensorDataset(x, y[:, 1].to(torch.double))


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
        return TorchTransformer.execute(self.circ, out)


class ModelQuantumNeuralNetwork(nn.Module):
    def __init__(self, circ: QuantumNeuralNetwork):
        super().__init__()
        self.circ = circ

    def forward(self, features):
        return TorchTransformer.execute(self.circ, features)


class ModelQuantumNeuralNetworkNative(nn.Module):
    """Test execution of qnn()"""

    def __init__(self, qnn: QuantumNeuralNetwork):
        super().__init__()
        self.qnn = qnn

    def forward(self, features):
        return self.qnn(features)


class TestLayers:
    circ = QuantumCircuit(2)
    theta = [Parameter(f"theta_{i}", 0.1) for i in range(3)]
    circ.x(0)
    circ.rx(0, theta[0])
    circ.ry(1, theta[1])
    circ.ry(0, theta[2])

    def _model_grad(self, model, batch_size):
        """Test one forward pass and gradient calculation of a model"""

        # TODO(zhaoyilun): Make out dimension configurable
        features = torch.randn(
            batch_size, 2, requires_grad=True, dtype=torch.double
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
        encoder_layer = AngleEmbedding(np.random.random((2,)), 2)
        qnn = QuantumNeuralNetwork(2, encoder_layer)
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
        qnn = QuantumNeuralNetwork(2, entangle_layer, backend="ScQ-P10")
        qnn.measure([0, 1], [0, 1])
        batch_size = 1

        # New invokation style
        model = ModelQuantumNeuralNetworkNative(qnn)
        self._model_grad(model, batch_size)

    def test_module_wrapper(self):
        weights = np.random.randn(2, 2)
        entangle_layer = BasicEntangleLayers(weights, 2)
        qnn = QuantumNeuralNetwork(2, entangle_layer)
        qnn.measure([0, 1], [0, 1])

        qlayer = ModuleWrapper(qnn)
        params = qlayer.parameters()

        assert np.allclose(
            qlayer.weights.detach().numpy(), params.__next__().detach().numpy()
        )

    def test_classify_random_dataset_quantum(self, num_epochs, batch_size):
        """Test a pure quantum nn training using a synthetic dataset

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
        encoder_layer = AngleEmbedding(np.random.random((2,)), num_qubits=2)
        entangle_layer = BasicEntangleLayers(weights, 2)
        qnn = QuantumNeuralNetwork(num_qubits, encoder_layer + entangle_layer)

        # Create hybrid model
        model = ModuleWrapper(qnn)

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

    def test_classify_random_dataset_hybrid(self, num_epochs, batch_size):
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
        qnn = QuantumNeuralNetwork(num_qubits, entangle_layer)
        qnn_model = ModelStandardCircuit(qnn)

        # Create MLP
        mlp = MLP(num_inputs, 4, 4)

        # Create hybrid model
        model = nn.Sequential(mlp, qnn_model)

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

    def test_runtime_qnn_construction(self, num_epochs, batch_size):
        """Test a pure quantum nn training using a synthetic dataset

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
        encoder_layer = AngleEmbedding(np.random.random((2,)), num_qubits=2)
        entangle_layer = BasicEntangleLayers(weights, 2)
        qnn = QuantumNeuralNetwork(num_qubits, [encoder_layer, entangle_layer])

        # Create hybrid model
        model = ModuleWrapper(qnn)

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

    def test_emplitude_embedding(self, num_epochs, batch_size):
        """Test a pure quantum nn training using a synthetic dataset

        Args:
            num_epochs: number of epoches for training
            batch_size: batch size for training

        """
        # Define the hyperparameters
        num_inputs = 2
        learning_rate = 0.01

        # Generate the dataset
        dataset = _generate_random_dataset(num_inputs, 100, one_hot=False)

        # Create QNN
        num_qubits = 1
        weights = np.random.randn(num_qubits, num_qubits)
        encoder_layer = AmplitudeEmbedding(
            np.random.random((2,)), num_qubits=num_qubits
        )
        entangle_layer = BasicEntangleLayers(weights, num_qubits)
        qnn = QuantumNeuralNetwork(num_qubits, [encoder_layer, entangle_layer])

        # Create hybrid model
        model = ModuleWrapper(qnn)

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
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
                # Forward pass
                outputs = model(inputs)
                # Since outputs are 1D, use rounding to classify (assuming binary classification, 0 or 1)
                predicted = (
                    outputs > 0.5
                ).int()  # Convert probabilities to binary predictions
                total += labels.size(0)
                correct += (
                    (predicted.squeeze() == labels.int()).sum().item()
                )  # Compare predictions with labels

        print(f"Accuracy: {100 * correct / total:.2f}%")
