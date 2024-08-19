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

import random


class Layout:
    """Layout between virtual qubits and physical qubits."""

    def __init__(self, mapping: dict = None):
        """mapping virtual qubits to physical qubits

        Args:
            mapping (dict): {virtual qubit: physical qubit, v0: p0, v1: p1, ...}
        """
        if mapping is not None:
            self.v2p = mapping
            self.p2v = {p: v for v, p in mapping.items()}
        else:
            self.p2v = {}
            self.v2p = {}

    def set_layout(self, layout):
        self.v2p = layout.v2p.copy()
        self.p2v = layout.p2v.copy()

    def generate_trivial_layout(self, virtual_qubits: int = None):
        """
        Args:
            virtual_qubits (int): the number of virtual qubits in a circuit
        """
        self.v2p = {k: k for k in range(virtual_qubits)}
        self.p2v = {k: k for k in range(virtual_qubits)}

    def generate_random_layout(
        self, virtual_qubits: int = None, physical_qubits: int = None
    ):
        """
        Args:
            virtual_qubits (int): the number of virtual qubits in a circuit
            physical_qubits (int): the number of physical qubits in a chip

        raise: The number of virtual qubits in the circuit cannot be greater than
                the number of physical qubits in the chip.

        """
        if virtual_qubits <= physical_qubits:
            virtual_qubit = random.sample(range(virtual_qubits), virtual_qubits)
            physical_qubit = random.sample(range(physical_qubits), virtual_qubits)
            for v, q in zip(virtual_qubit, physical_qubit):
                self.v2p[v] = q
            self.p2v = {p: v for v, p in self.v2p.items()}
        else:
            raise ValueError(
                "Error: The number of virtual qubits in the circuit cannot be greater than "
                "the number of physical qubits in the chip."
            )

    def from_v2p_dict(self, input_dict: dict = None):
        """
        Args:
            input_dict (dict): {virtual qubit: physical qubit, v0: p0, v1: p1, ...}
        """
        if input_dict is not None:
            self.v2p = input_dict
            self.p2v = {p: v for v, p in input_dict.items()}

    def from_p2v_dict(self, input_dict: dict = None):
        """
        Args:
            input_dict (dict): {physical qubit: virtual qubit, p0: v0, p1: v1, ...}
        """
        if input_dict is not None:
            self.p2v = input_dict
            self.v2p = {v: p for p, v in input_dict.items()}

    def swap(self, a, b):
        self.v2p[a], self.v2p[b] = self.v2p[b], self.v2p[a]
        self.p2v = {p: v for v, p in self.v2p.items()}
