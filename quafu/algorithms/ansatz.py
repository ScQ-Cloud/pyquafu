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
"""Ansatz circuits for VQA."""
from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np
from quafu.circuits.quantum_circuit import QuantumCircuit
from quafu.elements import Parameter, QuantumGate
from quafu.synthesis.evolution import ProductFormula

from .hamiltonian import Hamiltonian
from .interface_provider import InterfaceProvider
from .templates.base_embedding import BaseEmebdding


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
        self._h = hamiltonian
        self._num_layers = num_layers
        self._evol = ProductFormula()

        # Initialize parameters
        self._beta = np.array([Parameter(f"beta_{i}", 0.0) for i in range(num_layers)])
        self._gamma = np.array([Parameter(f"gamma_{i}", 0.0) for i in range(num_layers)])

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

    def update_params(self, paras_list: List[float]):
        """Update parameters of QAOA circuit"""
        # First build parameter list
        assert len(paras_list) == 2 * self._num_layers
        beta, gamma = paras_list[: self._num_layers], paras_list[self._num_layers :]
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
        self._theta = np.array([Parameter(f"theta_{i}", 0.0) for i in range((layer + 1) * num_qubits)])
        self._theta = np.reshape(self._theta, (layer + 1, num_qubits))
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


# pylint: disable=too-many-instance-attributes
class QuantumNeuralNetwork(Ansatz):
    """A Wrapper of quantum circuit as QNN"""

    # TODO(zhaoyilun): docs
    def __init__(self, num_qubits: int, layers: List[Any], interface="torch", backend="sim"):
        """"""
        # Get transformer according to specified interface
        self._transformer = InterfaceProvider.get(interface)
        self._layers = layers

        self._weights = None

        self._backend = backend

        self._num_tunable_params = 0

        if isinstance(layers[0], QuantumGate):
            self._legacy_if = True
        else:
            if isinstance(layers[0], BaseEmebdding):
                self._legacy_if = False
            else:
                raise TypeError(f"expect the first layer to be an embedding layer, but get a {type(layers[0])}")
        super().__init__(num_qubits)

    def _reset_circ(self):
        self.gates = []
        self.instructions = []
        self._variables = []
        self._parameter_grads = {}

    def is_legacy_if(self) -> bool:
        return self._legacy_if

    def reconstruct(self, inputs):
        """
        Args:
            inputs (List[float]): inputs to the qnn

        Notes:
            Before v0.4.3, only <list of gates> can be passed as `layers` to QuantumNeuralNetwork
            thus circuit structure is statically determined at creation time.
            However, we should allow circuit structure to be dynamically determined at runtime to
            accommodate embedding methods other than angle embedding
            thus this api is used by estimator to construct circuit at runtime
        """
        self._reset_circ()
        assert isinstance(self._layers[0], BaseEmebdding)
        circ = self._layers[0](inputs, self.num)
        for layer in self._layers[1:]:
            circ += layer
        self.add_gates(circ)
        self.get_parameter_grads()

    def __call__(self, inputs):
        """Compute outputs of QNN given input features"""
        # pylint: disable=import-outside-toplevel
        from quafu.algorithms.estimator import Estimator

        estimator = Estimator(self, backend=self._backend)
        return self._transformer.execute(self, inputs, estimator=estimator)

    def _build(self):
        """Essentially initialize weights using transformer"""

        if self.is_legacy_if():
            self.add_gates(self._layers)
        else:
            circ = self._layers[0]
            if len(self._layers) > 1:
                for layer in self._layers[1:]:
                    circ += layer
            self.add_gates(circ)

        self._weights = self._transformer.init_weights((1, self.num_tunable_parameters))

    @property
    def weights(self):
        return self._weights

    @property
    def num_tunable_parameters(self):
        """
        Return the number of tunable parameters

        Notes:
            We use _num_tunable_params to store the value not only for better efficiency but also
            for correctness because when updating parameters using parameter shift, the parameters become
            ParameterExpression and has no attribute `tunable`. So this value is only calculated at the first time
            and we assume it will be changed later.
        """
        if not self._num_tunable_params:
            for g in self.gates:
                paras = g.paras
                for p in paras:
                    if hasattr(p, "tunable") and p.tunable:
                        self._num_tunable_params += 1
        return self._num_tunable_params
