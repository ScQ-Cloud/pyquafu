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
"""DAG Drawer."""

from typing import Any, Union

import graphviz

from ..circuits import QuantumCircuit
from ..dagcircuits.circuit_dag import circuit_to_dag
from ..dagcircuits.instruction_node import InstructionNode


def _extract_node_info(node):
    if isinstance(node, InstructionNode):
        name, label = str(id(node)), str(node)
    else:
        if not (node == -1 or node == float("inf")):
            raise ValueError("node should be -1 or float('inf')")
        name, label = str(node), str(node)
    return name, label


# TODO: complete docstring, test supports for notebook
def draw_dag(
    qc: Union[QuantumCircuit, None],
    dag: Any = None,
    output_format: str = "pdf",
    output_filename: str = "DAG",
):
    """
    Helper function to visualize the DAG.

    Args:
        qc (QuantumCircuit): pyquafu quantum circuit if provided
        dag (DAG): DAG object with nodes and edges, built from qc if not provided
        output_format (str): output format, including "png", "svg", "pdf"...
        output_filename (str): file name of generated image

    Returns:
        dot: graphviz.Digraph object
    """
    if dag is None:
        if qc is None:
            raise ValueError("qc must be provided if dag is not provided")
        dag = circuit_to_dag(qc)

    dot = graphviz.Digraph(filename=output_filename)

    for node in dag.nodes:
        name, label = _extract_node_info(node)
        dot.node(name, label=label)

    for edge in dag.edges(data=True):
        node1, node2, link = edge
        name1, _ = _extract_node_info(node1)
        name2, _ = _extract_node_info(node2)
        dot.edge(name1, name2, label=link["label"])

    dot.render(format=output_format, cleanup=True)
    return dot
