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
"""Ansatz circuits for VQA"""
from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np
from quafu.circuits.quantum_circuit import QuantumCircuit
from quafu.synthesis.evolution import ProductFormula

from .hamiltonian import Hamiltonian
from .interface_provider import InterfaceProvider


class Ansatz(QuantumCircuit, ABC):
    """Ansatz interface"""

    def __init__(self, num: int, *args, **kwargs):
        super().__init__(num, *args, **kwargs)
        self._build()

    @property
    def num_parameters(self):
        """Get the number of parameters"""
        return len(super().parameterized_gates)

    @abstractmethod
    def _build(self):
        pass


class QAOAAnsatz(Ansatz):
    """QAOA Ansatz"""

    def __init__(self, hamiltonian: Hamiltonian, num_qubits: int, num_layers: int = 1):
        """Instantiate a QAOAAnsatz"""
        # self._pauli_list = hamiltonian.pauli_list
        # self._coeffs = hamiltonian.coeffs
        self._h = hamiltonian
        self._num_layers = num_layers
        self._evol = ProductFormula()

        # Initialize parameters
        self._beta = np.zeros(num_layers)
        self._gamma = np.zeros(num_layers)

        # Build circuit structure
        super().__init__(num_qubits)

    @property
    def num_parameters(self):
        return len(self._beta) + len(self._gamma)

    @property
    def parameters(self):
        """Return complete parameters of the circuit"""
        paras_list = []
        for layer in range(self._num_layers):
            for _ in range(len(self._h.paulis)):
                paras_list.append(self._gamma[layer])
            for _ in range(self.num):
                paras_list.append(self._beta[layer])
        return paras_list

    def _add_superposition(self):
        """Apply H gate on all qubits"""
        for i in range(self.num):
            self.h(i)

    def _build(self):
        """Construct circuit"""

        self._add_superposition()

        for layer in range(self._num_layers):
            # Add H_D layer
            for pauli in self._h.paulis:
                gate_list = self._evol.evol(pauli, self._gamma[layer])
                for g in gate_list:
                    self.add_ins(g)

            # Add H_B layer
            for i in range(self.num):
                self.rx(i, self._beta[layer])

    def update_params(self, params: List[float]):
        """Update parameters of QAOA circuit"""
        # First build parameter list
        assert len(params) == 2 * self._num_layers
        beta, gamma = params[: self._num_layers], params[self._num_layers :]
        num_para_gates = len(self.parameterized_gates)
        assert num_para_gates % self._num_layers == 0
        self._beta, self._gamma = beta, gamma
        super().update_params(self.parameters)


class AlterLayeredAnsatz(Ansatz):
    """A type of quantum circuit template that
    are problem-independent and hardware efficient

    Reference:
        *Alternating layered ansatz*
        - http://arxiv.org/abs/2101.08448
        - http://arxiv.org/abs/1905.10876
    """

    def __init__(self, num_qubits: int, layer: int):
        """
        Args:
            num_qubits: Number of qubits.
            layer: Number of layers.
        """
        self._layer = layer
        self._theta = np.zeros((layer + 1, num_qubits))
        super().__init__(num_qubits)

    def _build(self):
        """Construct circuit.

        Apply `self._layer` blocks, each block consists of a rotation gates on each qubit
        and an entanglement layer.
        """
        cnot_pairs = [(i, (i + 1) % self.num) for i in range(self.num)]
        for layer in range(self._layer):
            for qubit in range(self.num):
                self.ry(qubit, self._theta[layer, qubit])
            for ctrl, targ in cnot_pairs:
                self.cnot(ctrl, targ)
        for qubit in range(self.num):
            self.ry(qubit, self._theta[self._layer, qubit])


class QuantumNeuralNetwork(Ansatz):
    """A Wrapper of quantum circuit as QNN"""

    # TODO(zhaoyilun): docs
    def __init__(
        self, num_qubits: int, layers: List[Any], interface="torch", backend="sim"
    ):
        """"""
        # Get transformer according to specified interface
        self._transformer = InterfaceProvider.get(interface)
        self._layers = layers

        # FIXME(zhaoyilun): don't use this default value
        self._weights = np.empty((1, 1))

        self._backend = backend
        super().__init__(num_qubits)

    def __call__(self, features):
        """Compute outputs of QNN given input features"""
        from .estimator import Estimator

        estimator = Estimator(self, backend=self._backend)
        return self._transformer.execute(self, features, estimator=estimator)

    def _build(self):
        """Essentially initialize weights using transformer"""
        self.add_gates(self._layers)

        self._weights = self._transformer.init_weights((1, self.num_parameters))

    @property
    def weights(self):
        return self._weights
