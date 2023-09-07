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

from quafu.circuits.quantum_circuit import QuantumCircuit
from quafu.synthesis.evolution import ProductFormula


class QAOACircuit(QuantumCircuit):
    """QAOA circuit"""

    def __init__(self, pauli: str, num_layers: int = 1):
        """Instantiate a QAOAAnsatz"""
        num_qubits = len(pauli)
        super().__init__(num_qubits)
        self._num_layers = num_layers
        self._evol = ProductFormula()
        self._build(pauli)

    def _build(self, pauli):
        """Construct circuit"""
        gate_list = self._evol.evol(pauli, 0.0)
        for g in gate_list:
            self.add_gate(g)

    # def get_expectations(self):
    #     """Calculate the expectations of an operator"""
    #     pass
