from .qfasm_parser import QfasmParser, QregNode
from quafu.dagcircuits.circuit_dag import node_to_gate
from quafu.dagcircuits.instruction_node import InstructionNode
from quafu.circuits.quantum_circuit import QuantumCircuit


def qasm_to_circuit(qasm):
    parser = QfasmParser()     
    nodes = parser.parse(qasm)

    n = 0
    gates = []
    measures = {}
    for node in nodes:
        if isinstance(node, QregNode):
            n = node.n
        if isinstance(node, InstructionNode):
            if node.name == "measure":
                for q, c in zip(node.pos.keys(), node.pos.values()):
                    measures[q] = c
            else:
                gates.append(node_to_gate(node))

    q = QuantumCircuit(n)
    q.gates = gates
    q.openqasm = qasm
    q.measures = measures
    return q

