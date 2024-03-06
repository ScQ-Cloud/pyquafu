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
"""Angel Embedding in Quantum Data embedding"""
import numpy as np
import quafu.elements.element_gates as qeg
from quafu.elements import QuantumGate

ROT = {"X": qeg.RXGate, "Y": qeg.RYGate, "Z": qeg.RZGate}


class AngleEmbedding:
    def __init__(self, features, num_qubits: int, rotation="X"):
        """
        Args:
            features(np.array): The data to be embedded
            num_qubits(int): the number of qubit
            rotation(str): type of rotations used
        """
        if rotation not in ROT:
            raise ValueError(f"Rotation option {rotation} not recognized.")
        if features.ndim == 1:
            self.features = features.reshape(1, -1)
        else:
            self.features = features
        self.batch_size, n_features = self.features.shape
        if n_features != num_qubits:
            raise ValueError("The length of Features must match num_qubits")
        self.num_qubits = num_qubits
        self.op = ROT[rotation]
        """Build the embedding circuit and get the gate_list"""
        self.gate_list = self._build()

    def _build(self):
        gate_list = []
        for j in range(self.batch_size):
            for i in range(self.num_qubits):
                gate = self.op(i, self.features[j, i])
                gate_list.append(gate)
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
