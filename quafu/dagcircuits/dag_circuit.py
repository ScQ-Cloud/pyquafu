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
"""DAG Circuit."""

from typing import Dict

import networkx as nx
from networkx.classes.multidigraph import MultiDiGraph
from quafu.dagcircuits.instruction_node import InstructionNode


class DAGCircuit(MultiDiGraph):
    """
    A directed acyclic graph (DAG) representation of a quantum circuit.

    Inherits from MultiDiGraph, which is a directed graph class that can store multiedges.
    Each node in the DAGCircuit represents a quantum operation, and each edge represents a
    dependency between operations. The DAGCircuit is used to optimize the circuit by
    identifying and merging common subcircuits.
    """

    def __init__(
        self, qubits_used=None, cbits_used=None, incoming_graph_data=None, **attr
    ):
        """
        Create a DAGCircuit.

        Args:
            qubits_used (set[int]): A set of qubits used in the circuit. Defaults to None.
            cbits_used (set[int]): A set of classical bits used in the circuit. Defaults to None.
            circuit_qubits (int): the number of qubits in the circuit. Defaults to None.
            incoming_graph_data: Data to initialize graph. If None (default) an empty graph is created.
            **attr: Keyword arguments passed to the MultiDiGraph constructor.

        Attributes:
            qubits_used (set[int]): A set of qubits used in the circuit.
            cbits_used (set[int]): A set of classical bits used in the circuit.
            num_instruction_nodes (int): The number of instruction nodes in the circuit.
        """
        super().__init__(incoming_graph_data, **attr)

        if qubits_used is None:
            self.qubits_used = set()
        elif isinstance(qubits_used, set):
            self.qubits_used = qubits_used
        else:
            raise ValueError("qubits_used should be a set or None")

        if cbits_used is None:
            self.cbits_used = set()
        elif isinstance(cbits_used, set):
            self.cbits_used = cbits_used
        else:
            raise ValueError("cbits_used should be a set or None")

        self.circuit_qubits = None

        # num of instruction nodes
        self.num_instruction_nodes = 0

    # add new methods or override existing methods here.

    def update_circuit_qubits(self, circuit_qubits):
        """
        Number of qubits in quantum circuit.

        return:
            circuit_qubits: set of qubits in DAGCircuit
        """
        self.circuit_qubits = circuit_qubits
        return self.circuit_qubits

    def update_qubits_used(self):
        """
        qubits_used is a set of qubits used in DAGCircuit
        based on node -1's edges' labels, the qubits is the integer part of the label

        return:
            qubits_used: set of qubits used in DAGCircuit
        """
        if -1 not in self.nodes:
            raise ValueError("-1 should be in DAGCircuit, please add it first")
        self.qubits_used = {
            int(edge[2]["label"][1:]) for edge in self.out_edges(-1, data=True)
        }
        return self.qubits_used

    def update_cbits_used(self):
        """
        cbits_used is a set of cbits used in DAGCircuit
        calculated by  measure_node's cbits
        return:
            cbits_used: set of cbits used in DAGCircuit
        """
        for node in self.nodes:
            # if node.has an attribute 'name' and node.name == 'measure'
            if hasattr(node, "name") and node.name == "measure":
                self.cbits_used = set(node.pos.values())
        return self.cbits_used

    def update_num_instruction_nodes(self):
        """
        num_instruction_nodes is the number of instruction nodes in DAGCircuit
        calculated by len(self.nodes) - 2
        return:
            num_instruction_nodes: int

        """
        if -1 not in self.nodes:
            raise ValueError("-1 should be in DAGCircuit, please add it first")
        if float("inf") not in self.nodes:
            raise ValueError(
                'float("inf") should be in DAGCircuit, please add it first'
            )
        self.num_instruction_nodes = len(self.nodes) - 2

        return self.num_instruction_nodes

    def nodes_labels_resorted(self):
        """
        update nodes' labels
        This is only for convenience and does not affect the DAGCircuit itself.

        return:
            new_dag: DAGCircuit with nodes' labels resorted
        """

        # we need to make a new DAGCircuit
        new_dag = DAGCircuit()

        # watch out we should not only resort the nodes, but also the edges
        sorted_nodes_list = self.nodes_list()
        i = 0
        for node in sorted_nodes_list:
            if node.name != "measure":
                node.label = i
                # add the node in the end of new_dag
                new_dag.add_instruction_node_end(node)
                i += 1
            else:
                new_dag.add_instruction_node_end(node)
        return new_dag

    def nodes_dict(self):
        """
        nodes_dict is a dictionary of nodes with the node label as key and the node as value.
        without  -1 and float('inf')

        return:
            nodes_dict:  {node.label: node}
        """
        nodes_dict = {}
        for node in nx.topological_sort(self):
            if node != -1 and node != float("inf"):
                nodes_dict[node.label] = node
        return nodes_dict

    def nodes_list(self):
        """
        nodes_list is a list of nodes without  -1 and float('inf')

        return:
            nodes_list:  [node]
        """
        nodes_list = []
        for node in nx.topological_sort(self):
            if node != -1 and node != float("inf"):
                nodes_list.append(node)
        return nodes_list

    def node_qubits_predecessors(self, node: InstructionNode):
        """
        node_qubits_predecessors is a dict of {qubits -> predecessors }of node
        Args:
            node in DAGCircuit, node should not be -1
        Returns:
            node_qubits_predecessors: dict of {qubits -> predecessors }of node
        """
        if node not in self.nodes:
            raise ValueError("node should be in DAGCircuit")
        if node in [-1]:
            raise ValueError("-1 has no predecessors")

        predecessor_nodes = [edge[0] for edge in self.in_edges(node, data=True)]
        qubits_labels = [
            int(edge[2]["label"][1:]) for edge in self.in_edges(node, data=True)
        ]
        return dict(zip(qubits_labels, predecessor_nodes))

    def node_qubits_successors(self, node: InstructionNode):
        """
        node_qubits_successors is a dict of {qubits -> successors }of node
        Args:
            node in DAGCircuit, node should not be  float('inf')
        Returns:
            node_qubits_successors: dict of {qubits -> successors }of node


        """
        if node not in self.nodes:
            raise ValueError("node should be in DAGCircuit")
        if node in [float("inf")]:
            raise ValueError('float("inf") has no successors')
        successor_nodes = [edge[1] for edge in self.out_edges(node, data=True)]
        qubits_labels = [
            int(edge[2]["label"][1:]) for edge in self.out_edges(node, data=True)
        ]
        return dict(zip(qubits_labels, successor_nodes))

    def node_qubits_inedges(self, node: InstructionNode):
        """
        node_qubits_inedges is a dict of {qubits -> inedges }of node
        Args:
            node in DAGCircuit, node should not be -1
        Returns:
            node_qubits_inedges: dict of {qubits -> inedges }of node
        """
        if node not in self.nodes:
            raise ValueError("node should be in DAGCircuit")
        if node in [-1]:
            raise ValueError("-1 has no predecessors")

        inedges = list(self.in_edges(node, data=True, keys=True))  # we can get u,v,k,d
        qubits_labels = [
            int(edge[2]["label"][1:]) for edge in self.in_edges(node, data=True)
        ]
        return dict(zip(qubits_labels, inedges))

    def node_qubits_outedges(self, node: InstructionNode):
        """
        node_qubits_outedges is a dict of {qubits -> outedges }of node
        Args:
            node in DAGCircuit, node should not be float('inf')
        Returns:
            node_qubits_outedges: dict of {qubits -> outedges }of node
        """
        if node not in self.nodes:
            raise ValueError("node should be in DAGCircuit")
        if node in [float("inf")]:
            raise ValueError('float("inf") has no successors')
        outedges = list(
            self.out_edges(node, data=True, keys=True)
        )  # we can get u,v,k,d
        qubits_labels = [
            int(edge[2]["label"][1:]) for edge in self.out_edges(node, data=True)
        ]
        return dict(zip(qubits_labels, outedges))

    def remove_instruction_node(self, gate: InstructionNode):
        """
        remove a gate from DAGCircuit, and all edges connected to it.
        add new edges about qubits of removed gate between all predecessors and successors of removed gate.
        Args:
            gate: InstructionNode, gate should be in DAGCircuit, gate should not be -1 or float('inf')
        """

        if gate not in self.nodes:
            raise ValueError("gate should be in DAGCircuit")
        if gate in [-1, float("inf")]:
            raise ValueError('gate should not be -1 or float("inf")')

        qubits_predecessors = self.node_qubits_predecessors(gate)
        qubits_successors = self.node_qubits_successors(gate)
        for qubit in gate.pos:
            if qubits_predecessors[qubit] != -1 and qubits_successors[qubit] != float(
                "inf"
            ):
                self.add_edge(
                    qubits_predecessors[qubit],
                    qubits_successors[qubit],
                    label=f"q{qubit}",
                )
            elif qubits_predecessors[qubit] == -1 and qubits_successors[qubit] != float(
                "inf"
            ):
                self.add_edge(
                    qubits_predecessors[qubit],
                    qubits_successors[qubit],
                    label=f"q{qubit}",
                    color="green",
                )
            else:
                self.add_edge(
                    qubits_predecessors[qubit],
                    qubits_successors[qubit],
                    label=f"q{qubit}",
                    color="red",
                )

        self.remove_node(gate)

        # update qubits
        self.update_qubits_used()

    # pylint: disable=inconsistent-return-statements
    def merge_dag(self, other_dag):
        """
        merge other_dag into self
        Args:
            other_dag: DAGCircuit
        """
        if not isinstance(other_dag, DAGCircuit):
            raise ValueError("other_dag should be a DAGCircuit")
        if other_dag is None:
            return self
        if self is None:
            return other_dag

        # for the same qubits (intersection),
        # remove the outgoing edges from the final node of the original DAG
        # and the incoming edges from the initial node of the other DAG,
        # then connect the corresponding tail and head nodes by adding edges
        other_dag_qubits_used = other_dag.update_qubits_used()
        self_qubits_used = self.update_qubits_used()

        insect_qubits = self_qubits_used & other_dag_qubits_used
        end_edges_labels_1 = self.node_qubits_inedges(float("inf"))
        start_edges_labels_2 = other_dag.node_qubits_outedges(-1)

        if len(insect_qubits) != 0:
            for insect_qubit in insect_qubits:
                self.remove_edge(end_edges_labels_1[insect_qubit][:3])  # pylint: disable=no-value-for-parameter
                other_dag.remove_edge(start_edges_labels_2[insect_qubit][:3])
                self.add_edge(
                    end_edges_labels_1[insect_qubit][0],
                    start_edges_labels_2[insect_qubit][1],
                    label=f"q{insect_qubit}",
                )

        # add other_dag's nodes and edges into self
        # ！if we add edges, we don't need to add nodes again
        self.add_nodes_from(other_dag.nodes(data=True))
        self.add_edges_from(other_dag.edges(data=True))

        # update qubits
        self.update_qubits_used()
        return  # noqa:502

    def add_instruction_node(
        self,
        gate: InstructionNode,
        predecessors_dict: Dict[int, InstructionNode],
        successors_dict: Dict[int, InstructionNode],
    ):
        """
        add an instruction node into DAGCircuit, and all edges connected to it.
        add new edges about qubits of new gate between all predecessors and successors of new gate.
        Args:
            gate: InstructionNode, gate should not be -1 or float('inf')
            predecessors_dict: dict of {qubits -> predecessors }of gate
            successors_dict: dict of {qubits -> successors }of gate
        """
        if gate in [-1, float("inf")]:
            raise ValueError('gate should not be -1 or float("inf")')

        # remove the edges between the predecessors, successors about the qubits used by the added node
        qubits_pre_out_edges = []
        qubits_suc_in_edges = []
        self.update_qubits_used()

        for qubit in gate.pos:
            if qubit in self.qubits_used:
                pre_out_edges = self.node_qubits_outedges(predecessors_dict[qubit])
                qubits_pre_out_edges.append(
                    (pre_out_edges[qubit][:3])
                )  # use [:3] to get the key of the edge:u,v,k.

                suc_in_edges = self.node_qubits_inedges(successors_dict[qubit])
                qubits_suc_in_edges.append((suc_in_edges[qubit][:3]))

        qubits_removed_edges = set(qubits_pre_out_edges) | set(qubits_suc_in_edges)
        self.remove_edges_from(qubits_removed_edges)

        # add the new node and edges
        self.add_node(gate, color="blue")
        for qubit in gate.pos:
            if predecessors_dict[qubit] == -1:
                self.add_edge(
                    predecessors_dict[qubit], gate, label=f"q{qubit}", color="green"
                )
            else:
                self.add_edge(predecessors_dict[qubit], gate, label=f"q{qubit}")
            if successors_dict[qubit] == float("inf"):
                self.add_edge(
                    gate, successors_dict[qubit], label=f"q{qubit}", color="red"
                )
            else:
                self.add_edge(gate, successors_dict[qubit], label=f"q{qubit}")

        # update qubits
        self.update_qubits_used()

    def add_instruction_node_end(self, gate: InstructionNode):
        """
        add an instruction node at the end of DAGCircuit,while before float('inf') node.
        Args:
            gate: InstructionNode, gate should not be -1 or float('inf')
        """

        if gate in [-1, float("inf")]:
            raise ValueError('gate should not be -1 or float("inf")')
        if -1 not in self.nodes and float("inf") not in self.nodes:
            # if DAGCircuit is empty, add -1 and float('inf') first
            self.add_nodes_from([(-1, {"color": "green"})])
            self.add_nodes_from([(float("inf"), {"color": "red"})])

            # add the new node and edges
            self.add_node(gate, color="blue")
            for qubit in gate.pos:
                self.add_edge(-1, gate, label=f"q{qubit}", color="green")
                self.add_edge(gate, float("inf"), label=f"q{qubit}", color="red")
        elif -1 in self.nodes and float("inf") in self.nodes:
            # get the predecessors_dict of the new node
            inf_predecessors_dict = self.node_qubits_predecessors(float("inf"))
            # if some qubits of the new node is not in inf_predecessors_dict.keys(),
            # we need to add the gate_predecessors_dict[qubit] = -1
            gate_predecessors_dict = {}
            for qubit in gate.pos:
                if qubit not in inf_predecessors_dict.keys():
                    gate_predecessors_dict[qubit] = -1
                else:
                    gate_predecessors_dict[qubit] = inf_predecessors_dict[qubit]
            # and the successor of the new node is float('inf')
            gate_successors_dict = {qubit: float("inf") for qubit in gate.pos}
            # add the new node and edges using add_instruction_node method
            self.add_instruction_node(
                gate, gate_predecessors_dict, gate_successors_dict
            )
        else:
            raise ValueError(
                'DAGCircuit should have -1 and float("inf") at the same time'
            )

    # pylint: disable=inconsistent-return-statements
    def substitute_node_with_dag(self, gate: InstructionNode, input_dag):
        """
        substitute node with input_dag
        Args:
            gate: InstructionNode
            input_dag: DAGCircuit
        """

        if gate not in self.nodes:
            raise ValueError("node should be in DAGCircuit")
        if gate in [-1, float("inf")]:
            raise ValueError('node should not be -1 or float("inf")')
        if not isinstance(input_dag, DAGCircuit):
            raise ValueError("input_dag should be a DAGCircuit")
        if input_dag is None:
            self.remove_instruction_node(gate)
            return  # noqa:502
        if self is None:
            return input_dag

        # qubits set of input_dag should be the same as gate‘s qubits
        if input_dag.update_qubits_used() != set(gate.pos):
            raise ValueError(
                "qubits set of input_dag should be the same as the gate‘s qubits"
            )

        # Find all predecessors and successors of the node to be replaced
        predecessors_dict = self.node_qubits_predecessors(gate)
        successors_dict = self.node_qubits_successors(gate)

        # Remove the node and its edges from the dag graph
        self.remove_node(gate)

        # Add the input_dag into self:
        input_dag_startnodes = input_dag.node_qubits_successors(-1)
        input_dag_endnodes = input_dag.node_qubits_predecessors(float("inf"))

        input_dag_qubits = input_dag.update_qubits_used()

        input_dag.remove_node(-1)
        input_dag.remove_node(float("inf"))

        # Add edges between the nodes in input_dag and the predecessors and successors of the original node
        for qubit in input_dag_qubits:
            if predecessors_dict[qubit] == -1:
                self.add_edge(
                    predecessors_dict[qubit],
                    input_dag_startnodes[qubit],
                    label=f"q{qubit}",
                    color="green",
                )
            else:
                self.add_edge(
                    predecessors_dict[qubit],
                    input_dag_startnodes[qubit],
                    label=f"q{qubit}",
                )
            if successors_dict[qubit] == float("inf"):
                self.add_edge(
                    input_dag_endnodes[qubit],
                    successors_dict[qubit],
                    label=f"q{qubit}",
                    color="red",
                )
            else:
                self.add_edge(
                    input_dag_endnodes[qubit], successors_dict[qubit], label=f"q{qubit}"
                )

        # Add nodes and edges from input_dag to self
        self.add_nodes_from(input_dag.nodes(data=True))
        self.add_edges_from(input_dag.edges(data=True))
        return  # noqa:502

    def is_dag(self) -> bool:
        """
        is_dag is a bool value to check if DAGCircuit is a DAG

        Returns:
            is_dag: bool. True if DAGCircuit is a DAG, False otherwise.
        """
        return nx.is_directed_acyclic_graph(self)

    def get_measure_nodes(self):
        """Get all 'measure' nodes in dag.

        Returns (list): measure nodes list in dag.
        """
        measure_nodes = []
        for node in self.nodes:
            if node not in (-1, float("inf")):
                if node.name == "measure":
                    measure_nodes.append(node)
        return measure_nodes

    def remove_measure_nodes(self, only_last_measure=False):
        """Remove measure nodes in dag.
        Args:
            only_last_measure (bool): If Fales, remove all measure nodes in dag;
                                      If True, remove the last measure node in dag.
        """
        measure_nodes = self.get_measure_nodes()
        if only_last_measure:
            self.remove_instruction_node(measure_nodes[-1])
        else:
            for node in measure_nodes:
                self.remove_instruction_node(node)
