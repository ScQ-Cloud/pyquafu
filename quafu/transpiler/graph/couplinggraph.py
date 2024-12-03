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
from quafu.transpiler.graph.circuitgraph import draw_graph


class CouplingGraph:
    def __init__(self, coupling_list=None):
        """CouplingGraph initializer

        Args:
            coupling_list (list): [[q0,q1,fidelity], [q1,q0,fidelity], ...]
        """
        self.graph = nx.DiGraph()  # the qubits coupling structure graph of a quantum chip
        self._distance_matrix = None  # adjacency matrix for coupling graph
        self._qubits_list = None  # list of physical qubits in the coupling graph
        self._num_qubits = None  # number of qubits in the coupling graph
        self.is_bidirectional = True  # Determine whether it is bidirectional

        if coupling_list is not None:
            # Sort nodes
            nodeset = sorted({item for sublist in coupling_list for item in sublist[0:2]})
            self.graph.add_nodes_from(nodeset)
            if len(coupling_list[0]) == 3:
                for edge in coupling_list:
                    self.graph.add_edges_from([(edge[0], edge[1], {"fidelity": edge[2]})])
            elif len(coupling_list[0]) == 2:
                for edge in coupling_list:
                    self.graph.add_edges_from([(edge[0], edge[1], {"fidelity": 1.0})])
            else:
                raise TypeError("Error: The input coupling_list is wrong.")

        for pair in coupling_list:
            reverse_pair = (pair[1], pair[0])
            if reverse_pair not in coupling_list:
                self.is_bidirectional = False
                break

        self._edge_dict = None
        self._path_fidelity = None

    @property
    def num_qubits(self):
        """Get the number of physical qubits in this coupling graph."""
        if self._num_qubits is None:
            self._num_qubits = len(self.graph)
        return self._num_qubits

    @property
    def edge_dict(self):
        """Get the edge weight dictionary for this graph.

        Returns:
            self._edge_dict (dict): {(qubit1, qubit2): fidelity, ...}
        """
        if self._edge_dict is None:
            self._edge_dict = nx.get_edge_attributes(self.graph, "fidelity")
        return self._edge_dict

    @property
    def qubits_list(self):
        """Get a list of physical qubits"""
        if self._qubits_list is None:
            self._qubits_list = list(self.graph.nodes())
        return self._qubits_list

    def subgraph(self, node_list):
        """Get the subgraph of this graph.

        Args:
            node_list: node list of subgraph.

        Returns:
            sub_coupling (CouplingGraph)
        """
        sub_coupling = CouplingGraph()
        sub_coupling.graph = self.graph.subgraph(node_list)
        return sub_coupling

    def is_connected(self):
        """Test if the graph is connected.

        Returns:
            nx.is_directed(self.graph) (bool):
            True if the nodes in the graph are connected, False otherwise
        """
        return nx.is_directed(self.graph)

    def neighbors(self, qubit):
        """Returns the nearest neighbors of a given physical qubit in this graph."""
        return self.graph.neighbors(qubit)

    @property
    def distance_matrix(self):
        """Get the distance matrix for the coupling graph."""
        if self._distance_matrix is None:
            if not self.is_connected():
                raise ValueError("Error: This coupling diagram is not connected.")
            self._distance_matrix = nx.floyd_warshall_numpy(self.graph)
        return self._distance_matrix

    def shortest_undirected_path(self, source_qubit, target_qubit):
        """Get the shortest path between source_qubit and target_qubit.

        Args:
            source_qubit: Starting node for path.
            target_qubit: Ending node for path.
        Returns:
            path: Shortest path include both the source and target in the path.
        """
        path = nx.shortest_path(self.graph, source=source_qubit, target=target_qubit)
        if not path:
            raise ValueError(f"Error: Nodes {source_qubit} and {target_qubit} are not connected.")
        return path

    @property
    def path_fidelity(self):
        """Get path fidelity

        The product of the fidelity of all two-qubit gates on the connection path of bits n1 and n2,
        that is, the sum of their respective logarithms.

        Returns: self._path_fidelity
        """
        if self._path_fidelity is None:
            self._path_fidelity = {}
            nodes = len(self.graph.nodes)
            for n1 in range(nodes - 1):
                for n2 in range(n1 + 1, nodes):
                    swap_path = nx.shortest_path(self.graph, n1, n2)
                    if len(swap_path) == 2:  # not need swap
                        self._path_fidelity[(n1, n2)] = np.log(self.edge_dict[(swap_path[0], swap_path[1])])
                    else:
                        fidelity = 0
                        for i in range(len(swap_path) - 2):  # SWAP gates need to be inserted
                            min_f = min(
                                np.log(self.edge_dict[(swap_path[i], swap_path[i + 1])]),
                                np.log(self.edge_dict[(swap_path[i + 1], swap_path[i])]),
                            )
                            max_f = max(
                                np.log(self.edge_dict[(swap_path[i], swap_path[i + 1])]),
                                np.log(self.edge_dict[(swap_path[i + 1], swap_path[i])]),
                            )
                            fidelity += 2 * max_f + min_f
                        fidelity += np.log(self.edge_dict[(swap_path[-1], swap_path[-2])])
                        self._path_fidelity[(n1, n2)] = fidelity

                    swap_path = nx.shortest_path(self.graph, n2, n1)
                    if len(swap_path) == 2:  # not need swap
                        self._path_fidelity[(n2, n1)] = np.log(self.edge_dict[(swap_path[0], swap_path[1])])
                    else:
                        fidelity = 0
                        for i in range(len(swap_path) - 2):  # SWAP gates need to be inserted
                            min_f = min(
                                np.log(self.edge_dict[(swap_path[i], swap_path[i + 1])]),
                                np.log(self.edge_dict[(swap_path[i + 1], swap_path[i])]),
                            )
                            max_f = max(
                                np.log(self.edge_dict[(swap_path[i], swap_path[i + 1])]),
                                np.log(self.edge_dict[(swap_path[i + 1], swap_path[i])]),
                            )
                            fidelity += 2 * max_f + min_f
                        fidelity += np.log(self.edge_dict[(swap_path[-1], swap_path[-2])])
                        self._path_fidelity[(n2, n1)] = fidelity

        return self._path_fidelity

    def do_bidirectional(self):
        """Convert unidirectional edges to bidirectional edges."""
        edges_label = self.graph.edges(data=True)
        edges = self.graph.edges()
        for e1, e2, label in edges_label:
            if (e2, e1) not in edges:
                self.graph.add_edges_from([(e2, e1, label)])

        self.is_bidirectional = True

    def draw(self):
        """Draws the coupling map."""
        draw_graph(self.graph)
