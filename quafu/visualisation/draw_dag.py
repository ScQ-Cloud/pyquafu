import graphviz

from quafu import QuantumCircuit
from quafu.dagcircuits.circuit_dag import circuit_to_dag
from quafu.dagcircuits.instruction_node import InstructionNode


def _extract_node_info(node):
    if isinstance(node, InstructionNode):
        name, label = str(id(node)), node.name
    else:
        assert node == -1 or node == float("inf")
        name, label = str(node), str(node)
    return name, label


def draw_dag(qc: QuantumCircuit):
    dag = circuit_to_dag(qc)
    dot = graphviz.Digraph()

    for node in dag.nodes:
        name, label = _extract_node_info(node)
        dot.node(name, label=label)

    for edge in dag.edges(data=True):
        node1, node2, link = edge
        name1, label1 = _extract_node_info(node1)
        name2, label2 = _extract_node_info(node2)
        dot.edge(name1, name2, label=link['label'])

    dot.view(cleanup=True)
