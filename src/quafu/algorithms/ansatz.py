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
        gate_list = self._evol.evol(pauli, 0.)
        for g in gate_list:
            self.add_gate(g)

    # def get_expectations(self):
    #     """Calculate the expectations of an operator"""
    #     pass
