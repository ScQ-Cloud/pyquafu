

from .quantum_circuit import QuantumCircuit
from ..transpiler.Qcovercompiler import QcoverCompiler

class QAOACircuit(QuantumCircuit):
    def __init__(self, logical_qubits, physical_qubits, nodes, edges, params, p, gate="CNOT"):
        num = logical_qubits
        self.logical_qubits = logical_qubits
        self.physical_qubits = physical_qubits
        self.nodes = nodes
        self.edges = edges
        self.paras = params
        self.p = p
        self.gate = gate
        super().__init__(num)

    def compile_to_IOP(self):
        """
        QASM from qcover directly
        """
        qaoa_compiler = QcoverCompiler()
        self.qasm = qaoa_compiler.graph_to_qasm(self.logical_qubits, self.physical_qubits, self.nodes, self.edges, self.paras, self.p, gate=self.gate)

    def upate_paras(self, paras):
        self.paras = paras
        pass




