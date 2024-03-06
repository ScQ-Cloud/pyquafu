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

"""Layers consisting of one-parameter single-qubit rotations on each qubit, followed by a closed chain or ring of CNOT gates"""
import numpy as np
import quafu.elements.element_gates as qeg
from quafu.elements import QuantumGate
from quafu.elements.parameters import Parameter

ROT = {"X": qeg.RXGate, "Y": qeg.RYGate, "Z": qeg.RZGate}


class BasicEntangleLayers:
    def __init__(self, weights=None, num_qubits=None, repeat=None, rotation="X"):
        """
        Args:
            weights(array): Each weight is used as a parameter for the rotation
            num_qubits(int): the number of qubit
            rotation(str): one-parameter single-qubit gate to use
            repeat(int): the number of layers, only work while the weights is not provided
        """
        if num_qubits is None:
            raise ValueError(f"num_qubits must be provided")
        if weights is not None:
            weights = np.asarray(weights)
            shape = np.shape(weights)
            ##TODO(): If weights are batched, i.e. dim>3, additional processing is required
            if weights.ndim > 2:
                raise ValueError(f"Weights tensor must be 2-dimensional ")

            if not (
                len(shape) == 3 or len(shape) == 2
            ):  # 3 is when batching, 2 is no batching
                raise ValueError(
                    f"Weights tensor must be 2-dimensional "
                    f"or 3-dimensional if batching; got shape {shape}"
                )

            if shape[-1] != num_qubits:
                # index with -1 since we may or may not have batching in first dimension
                raise ValueError(
                    f"Weights tensor must have last dimension of length {num_qubits}; got {shape[-1]}"
                )
        else:
            self.weights = None
            if repeat is None:
                raise ValueError(f"repeat must be provided if weights is None")
        # convert weights to numpy array if weights is list otherwise keep unchanged
        self.weights = weights
        self.num_qubits = num_qubits
        self.op = ROT[rotation]
        self.repeat = repeat

        """Build the quantum basic_entangle layer and get the gate_list"""
        self.gate_list = self._build()

    def _build(self):
        gate_list = []
        if self.weights is not None:
            repeat = np.shape(self.weights)[-2]
            theta = [
                Parameter(
                    "theta_%d" % (layer * self.num_qubits + i), self.weights[layer][i]
                )
                for layer in range(repeat)
                for i in range(self.num_qubits)
            ]
        else:
            repeat = self.repeat
            theta = [
                Parameter("theta_%d" % j, np.round(np.random.rand(), 3))
                for j in range(repeat * self.num_qubits)
            ]
        for layer in range(repeat):
            j = layer * self.num_qubits
            for i in range(self.num_qubits):
                gate = self.op(i, theta[j])
                gate_list.append(gate)
                j += 1

            # if num_qubits equals two, it just need to apply CNOT one time
            if self.num_qubits == 2:
                gate_list.append(qeg.CXGate(0, 1))

            elif self.num_qubits > 2:
                for i in range(self.num_qubits):
                    gate_list.append(qeg.CXGate(i, (i + 1) % self.num_qubits))

        return gate_list

    def __iter__(self):
        return iter(self.gate_list)

    def __getitem__(self, index):
        return self.gate_list[index]

    def __add__(self, gates):
        """Addition operator."""
        out = []
        out.extend(self.gate_list)
        if all(isinstance(gate, QuantumGate) for gate in gates):
            out.extend(gates)
        else:
            raise TypeError("Contains unsupported gate")
        return out

    def __radd__(self, other):
        out = []
        if all(isinstance(gate, QuantumGate) for gate in other):
            out.extend(other)
        else:
            raise TypeError("Contains unsupported gate")
        out.extend(self.gate_list)
        return out
