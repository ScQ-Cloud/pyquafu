import networkx as nx
from typing import Dict, Any, List

from quafu.dagcircuits.instruction_node import InstructionNode

from networkx.classes.multidigraph import MultiDiGraph

class DAGCircuit(MultiDiGraph):
    def __init__(self,qubits_used=None, cbits_used=None, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
              
        if qubits_used is None:
            self.qubits_used = set()
        elif isinstance(qubits_used, set):
            self.qubits_used = qubits_used
        else:
            raise ValueError('qubits_used should be a set or None')

        if cbits_used is None:
            self.cbits_used = set()
        elif isinstance(cbits_used, set):
            self.cbits_used = cbits_used
        else:
            raise ValueError('cbits_used should be a set or None')
        
        # num of instruction nodes
        self.num_instruction_nodes = 0


    # add new methods or override existing methods here.

    def update_qubits_used(self):
        '''
        qubits_used is a set of qubits used in DAGCircuit
        based on node -1's edges' labels, the qubits is the integer part of the label

        return:
            qubits_used: set of qubits used in DAGCircuit
        '''
        if -1 not in self.nodes:
            raise ValueError('-1 should be in DAGCircuit, please add it first')
        self.qubits_used = set([int(edge[2]['label'][1:]) for edge in self.out_edges(-1, data=True)])
        return self.qubits_used
    
    def update_cbits_used(self):
        '''
        cbits_used is a set of cbits used in DAGCircuit
        calculated by  measurenode's cbits        
        return:
            cbits_used: set of cbits used in DAGCircuit
        '''
        for node in self.nodes:
        # if node.has a attribute 'name' and node.name == 'measure'
            if hasattr(node, 'name') and node.name == 'measure':
                self.cbits_used = set(node.pos.values())   
        return self.cbits_used
    
    def update_num_instruction_nodes(self):
        '''
        num_instruction_nodes is the number of instruction nodes in DAGCircuit
        '''
        if -1 not in self.nodes:
            raise ValueError('-1 should be in DAGCircuit, please add it first')
        if float('inf') not in self.nodes:
            raise ValueError('float("inf") should be in DAGCircuit, please add it first')
        self.num_instruction_nodes = len(self.nodes) - 2

        for node in self.nodes:
            if hasattr(node, 'name') and node.name == 'measure':
                self.num_instruction_nodes -= 1
        return self.num_instruction_nodes

    

    def nodes_dict(self):
        '''
        nodes_dict is a dictionary of nodes with the node label as key and the node as value.
        without  -1 and float('inf')
        '''
        nodes_dict = {}
        for node in nx.topological_sort(self):
            if node != -1 and node != float('inf'):
                nodes_dict[node.label] = node
        return nodes_dict
     

    def nodes_list(self):
        ''' 
        nodes_list is a list of nodes without  -1 and float('inf')
        ''' 
        nodes_list = []
        for node in nx.topological_sort(self):
            if node != -1 and node != float('inf'):
                nodes_list.append(node)
        return nodes_list
    
    def node_qubits_predecessors(self, node:InstructionNode):
        '''
        node_qubits_predecessors is a dict of {qubits -> predecessors }of node
        Args:
            node in DAGCircuit, node should not be -1
        Returns:
            node_qubits_predecessors: dict of {qubits -> predecessors }of node
        '''
        # for edge in self.in_edges(node, data=True):
        #     print(edge[0], edge[1], edge[2])

        if node not in self.nodes:
            raise ValueError('node should be in DAGCircuit')
        if node in [-1]:
            raise ValueError('-1 has no predecessors')

        predecessor_nodes = [edge[0] for edge in self.in_edges(node, data=True)]
        qubits_labels = [int(edge[2]['label'][1:]) for edge in self.in_edges(node, data=True)]
        node_qubits_predecessors = dict(zip(qubits_labels, predecessor_nodes))
        return node_qubits_predecessors
    
    def node_qubits_successors(self, node:InstructionNode):
        '''
        node_qubits_successors is a dict of {qubits -> successors }of node
        Args: 
            node in DAGCircuit, node should not be  float('inf')
        Returns:
            node_qubits_successors: dict of {qubits -> successors }of node


        '''
        if node not in self.nodes:
            raise ValueError('node should be in DAGCircuit')
        if node in [float('inf')]:
            raise ValueError('float("inf") has no successors')
        successor_nodes = [edge[1] for edge in self.out_edges(node, data=True)]
        qubits_labels = [int(edge[2]['label'][1:]) for edge in self.out_edges(node, data=True)]
        node_qubits_successors = dict(zip(qubits_labels, successor_nodes))
        return node_qubits_successors

    def node_qubits_inedges(self, node:InstructionNode):
        '''
        node_qubits_inedges is a dict of {qubits -> inedges }of node
        Args:
            node in DAGCircuit, node should not be -1
        Returns:
            node_qubits_inedges: dict of {qubits -> inedges }of node
        '''
        if node not in self.nodes:
            raise ValueError('node should be in DAGCircuit')
        if node in [-1]:
            raise ValueError('-1 has no predecessors')

        inedges = [edge for edge in self.in_edges(node)]
        qubits_labels = [int(edge[2]['label'][1:]) for edge in self.in_edges(node, data=True)]
        node_qubits_inedges = dict(zip(qubits_labels, inedges))
        return node_qubits_inedges
    
    def node_qubits_outedges(self, node:InstructionNode):
        '''
        node_qubits_outedges is a dict of {qubits -> outedges }of node
        Args:
            node in DAGCircuit, node should not be float('inf')
        Returns:
            node_qubits_outedges: dict of {qubits -> outedges }of node
        '''
        if node not in self.nodes:
            raise ValueError('node should be in DAGCircuit')
        if node in [float('inf')]:
            raise ValueError('float("inf") has no successors')
        outedges = [edge for edge in self.out_edges(node)]
        qubits_labels = [int(edge[2]['label'][1:]) for edge in self.out_edges(node, data=True)]
        node_qubits_outedges = dict(zip(qubits_labels, outedges))
        return node_qubits_outedges
    
    def remove_instruction_node(self, gate:InstructionNode):
        '''
        remove a gate from DAGCircuit, and all edges connected to it.
        add new edges about qubits of removed gate between all predecessors and successors of removed gate.
        Args:
            gate: InstructionNode, gate should be in DAGCircuit, gate should not be -1 or float('inf')
        '''

        if gate not in self.nodes:
            raise ValueError('gate should be in DAGCircuit')
        if gate in [-1, float('inf')]:
            raise ValueError('gate should not be -1 or float("inf")')

        qubits_predecessors = self.node_qubits_predecessors(gate)
        qubits_successors = self.node_qubits_successors(gate)
        for qubit in gate.pos:
            if qubits_predecessors[qubit] != -1 and qubits_successors[qubit] != float('inf'):
                self.add_edge(qubits_predecessors[qubit], qubits_successors[qubit], label=f'q{qubit}')
            elif qubits_predecessors[qubit] == -1 and qubits_successors[qubit] != float('inf'):
                self.add_edge(qubits_predecessors[qubit], qubits_successors[qubit], label=f'q{qubit}',color='green')
            else:
                self.add_edge(qubits_predecessors[qubit], qubits_successors[qubit], label=f'q{qubit}',color='red')

        self.remove_node(gate)

        # update qubits
        self.qubits_used = self.update_qubits_used()


    def merge_dag(self, other_dag):
        '''
        merge other_dag into self
        Args:
            other_dag: DAGCircuit
        Returns:
            self: DAGCircuit
        '''
        if not isinstance(other_dag, DAGCircuit):
            raise ValueError('other_dag should be a DAGCircuit')
        if other_dag == None:
            return self
        if self == None:
            return other_dag
        
        # for the same qubits (intersection), 
        # remove the outgoing edges from the final node of the original DAG and the incoming edges from the initial node of the other DAG,
        # then connect the corresponding tail and head nodes by adding edges
        other_dag_qubits_used = other_dag.update_qubits_used()
        self_qubits_used = self.update_qubits_used()

        insect_qubits = self_qubits_used & other_dag_qubits_used
        end_edges_labels_1 = self.node_qubits_inedges(float('inf'))
        start_edges_labels_2 = other_dag.node_qubits_outedges(-1)

        if len(insect_qubits) != 0:
            for insect_qubit in insect_qubits:
                self.remove_edges_from([end_edges_labels_1[insect_qubit]])
                other_dag.remove_edges_from([start_edges_labels_2[insect_qubit]])
                self.add_edge(end_edges_labels_1[insect_qubit][0], start_edges_labels_2[insect_qubit][1], label=f'q{insect_qubit}')
        
        # add other_dag's nodes and edges into self
        # ！if we add edges, we don't need to add nodes again
        self.add_edges_from(other_dag.edges(data=True))

        # remove the edges between -1 and float('inf')
        self.remove_edges_from([edge for edge in self.edges(data=True) if edge[0] == -1 and edge[1] == float('inf')])
 
        # update qubits 
        self.qubits_used = self.update_qubits_used()
  
    def add_instruction_node(self, gate:InstructionNode,predecessors_dict:Dict[int,InstructionNode],successors_dict:Dict[int,InstructionNode]):
        '''
        add a gate into DAGCircuit, and all edges connected to it.
        add new edges about qubits of new gate between all predecessors and successors of new gate.
        Args:
            gate: InstructionNode, gate should not be -1 or float('inf')
            predecessors_dict: dict of {qubits -> predecessors }of gate
            successors_dict: dict of {qubits -> successors }of gate
        '''
        if gate in [-1, float('inf')]:
            raise ValueError('gate should not be -1 or float("inf")')
        
        #remove the edges between the predessors,successors about the qubits used by the added node
        qubits_pre_out_edges = []
        qubits_suc_in_edges = []
        for qubit in gate.pos:
            pre_out_edges = self.node_qubits_outedges(predecessors_dict[qubit])
            qubits_pre_out_edges.append(pre_out_edges[qubit])

            suc_in_edges = self.node_qubits_inedges(successors_dict[qubit])
            qubits_suc_in_edges.append(suc_in_edges[qubit])

        self.remove_edges_from(qubits_pre_out_edges)
        self.remove_edges_from(qubits_suc_in_edges)

        # add the new node and edges
        for qubit in gate.pos:
            if predecessors_dict[qubit] == -1:
                self.add_edge(predecessors_dict[qubit], gate, label=f'q{qubit}',color='green')
            else:
                self.add_edge(predecessors_dict[qubit], gate, label=f'q{qubit}')
            if successors_dict[qubit] == float('inf'):
                self.add_edge(gate, successors_dict[qubit], label=f'q{qubit}',color='red')
            else:
                self.add_edge(gate, successors_dict[qubit], label=f'q{qubit}')

        # update qubits
        self.qubits_used = self.update_qubits_used()


    def is_dag(self):
        '''
        is_dag is a bool value to check if DAGCircuit is a DAG
        '''
        return nx.is_directed_acyclic_graph(self)