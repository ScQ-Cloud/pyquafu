"""Ansatz circuits for VQA"""

from quafu.circuits.quantum_circuit import QuantumCircuit


class QAOACircuit(QuantumCircuit):
    """QAOA circuit"""

    def __init__(self, num: int, num_layers: int=1):
        """Instantiate a QAOAAnsatz"""
        super().__init__(num)
        self._num_layers = num_layers

    def build(self):
        """Construct circuit"""

    def get_expectations(self):
        """Calculate the expectations of an operator"""
        pass
