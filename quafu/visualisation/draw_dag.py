from typing import Any, Union

import graphviz
from quafu.dagcircuits.circuit_dag import circuit_to_dag
from quafu.dagcircuits.instruction_node import InstructionNode

from quafu import QuantumCircuit


def _extract_node_info(node):
    if isinstance(node, InstructionNode):
        name, label = str(id(node)), str(node)
    else:
        assert node == -1 or node == float("inf")
        name, label = str(node), str(node)
    return name, label


def draw_dag(
    qc: Union[QuantumCircuit, None],
    dag: Any = None,
    output_format: str = "pdf",
    output_filename: str = "DAG",
):
    """
    TODO: complete docstring, test supports for notebook

    Helper function to visualize the DAG

    Args:
        qc (QuantumCircuit): pyquafu quantum circuit if provided
        dag (DAG): DAG object with nodes and edges, built from qc if not provided
        output_format (str): output format, including "png", "svg", "pdf"...
        output_filename (str): file name of generated image

    Returns:
        dot: graphviz.Digraph object
    """
    if dag is None:
        assert qc is not None
        dag = circuit_to_dag(qc)

    dot = graphviz.Digraph(filename=output_filename)

    for node in dag.nodes:
        name, label = _extract_node_info(node)
        dot.node(name, label=label)

    for edge in dag.edges(data=True):
        node1, node2, link = edge
        name1, label1 = _extract_node_info(node1)
        name2, label2 = _extract_node_info(node2)
        dot.edge(name1, name2, label=link["label"])

    dot.render(format=output_format, cleanup=True)
    return dot
