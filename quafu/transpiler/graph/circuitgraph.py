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

import networkx as nx
import numpy as np
from quafu import QuantumCircuit
from quafu.elements import Barrier, Measure


def relabel_graph(graph):
    """Standardize the graph using degrees as node weights.
    """
    for nodes in graph.nodes.data():
        graph.add_node(nodes[0], weight=graph.degree[nodes[0]])
    return graph


def circuit_to_graph(circuit: QuantumCircuit):
    """
    Convert the circuit to a weighted graph, with the edge weights
     being the corresponding number of two-qubit gates.
     The multi-qubit gate becomes a weighted graph,
     which is equivalent to the full connection of each qubit.
    """
    num_qubits = circuit.num
    temp = np.zeros((num_qubits, num_qubits))
    gates_list = circuit.gates
    for gate in gates_list:
        if gate.name not in [Barrier.name, Measure.name]:
            if isinstance(gate.pos, list) and len(gate.pos) >= 2:
                index = sorted(gate.pos)
                for i in range(len(index)):
                    for j in range(i + 1, len(index)):
                        temp[index[i]][index[j]] = temp[index[i]][index[j]] + 1
            # else:
            #     index = gate.pos
            #     temp[index][index] = temp[index][index] + 1
    graph = nx.Graph(temp)
    graph = relabel_graph(graph)
    return graph


def draw_graph(g):
    """Draw weighted graph
    """
    import matplotlib.pyplot as plt
    label_name = list(list(g.edges(data=True))[0][2].keys())[0]
    labels = nx.get_edge_attributes(g, label_name)
    pos = nx.spring_layout(g)
    # pos = nx.circular_layout(g)
    nx.draw_networkx(g, pos=pos, with_labels=True, node_color='r', node_size=200, edge_color='b',
                     width=1, font_size=10)
    if isinstance(g, nx.MultiDiGraph) or isinstance(g, nx.MultiGraph):
        pass
    else:
        nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=labels)
    plt.show()
