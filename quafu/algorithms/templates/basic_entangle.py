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
from quafu.circuits import QuantumCircuit

ROT = {"X": qeg.RXGate, "Y": qeg.RYGate, "Z": qeg.RZGate}


class BasicEntangleLayers:
    def __init__(self, weights, num_qubits, rotation="X"):
        """
        Args:
            weights(array): Each weight is used as a parameter for the rotation
            num_qubits(int): the number of qubit
            rotation(str): one-parameter single-qubit gate to use
        """
        weights = np.asarray(weights)
        # convert weights to numpy array if weights is list otherwise keep unchanged
        shape = np.shape(weights)

        ##TODO(qtzhuang): If weights are batched, i.e. dim>3, additional processing is required
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
        self.weights = weights
        self.num_qubits = num_qubits
        self.op = ROT[rotation]
        """Build the quantum basic_entangle layer and get the gate_list"""
        self.gate_list = self._build()

    def _build(self):
        repeat = np.shape(self.weights)[-2]
        gate_list = []
        for layer in range(repeat):
            for i in range(self.num_qubits):
                gate = self.op(pos=i, paras=self.weights[layer][i])
                gate_list.append(gate)

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
