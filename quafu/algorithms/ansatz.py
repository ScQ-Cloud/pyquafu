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
from typing import List
import numpy as np

from quafu.algorithms.hamiltonian import Hamiltonian
from quafu.circuits.quantum_circuit import QuantumCircuit
from quafu.synthesis.evolution import ProductFormula


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
    """QAOA circuit"""

    def __init__(self, hamiltonian: Hamiltonian, num_layers: int = 1):
        """Instantiate a QAOAAnsatz"""
        self._pauli_list = hamiltonian.pauli_list
        self._coeffs = hamiltonian.coeffs
        self._num_layers = num_layers
        self._evol = ProductFormula()

        # Initialize parameters
        self._beta = np.zeros(num_layers)
        self._gamma = np.zeros(num_layers)

        # Build circuit structure
        num_qubits = len(self._pauli_list[0])
        super().__init__(num_qubits)

    @property
    def num_parameters(self):
        return len(self._beta) + len(self._gamma)

    @property
    def parameters(self):
        """Return complete parameters of the circuit"""
        paras_list = []
        for layer in range(self._num_layers):
            for _ in range(len(self._pauli_list)):
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
            for pauli_str in self._pauli_list:
                gate_list = self._evol.evol(pauli_str, self._gamma[layer])
                for g in gate_list:
                    self.add_gate(g)

            # Add H_B layer
            for i in range(self.num):
                self.rx(i, self._beta[layer])

    def update_params(self, beta: List[float], gamma: List[float]):
        """Update parameters of QAOA circuit"""
        # First build parameter list
        assert len(beta) == self._num_layers and len(gamma) == self._num_layers
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
