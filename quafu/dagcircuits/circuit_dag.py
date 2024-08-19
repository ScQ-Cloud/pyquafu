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

import numpy as np
from quafu import QuantumCircuit

from quafu.elements.element_gates import *
from quafu.elements.element_gates.clifford import *
from quafu.elements.element_gates.pauli import *

from quafu.elements import Barrier, Delay, XYResonance

import networkx as nx
from typing import Any, List
import copy

from quafu.dagcircuits.instruction_node import InstructionNode  # instruction_node.py in the same folder as circuit_dag.py now
from quafu.dagcircuits.dag_circuit import DAGCircuit  # dag_circuit.py in the same folder as circuit_dag.py now

# import pygraphviz as pgv
from networkx.drawing.nx_pydot import write_dot
from IPython.display import Image, SVG


# transform a gate in quantumcircuit of quafu(not include measure_gate),
# into a node in the graph, with specific label.
def gate_to_node(input_gate, specific_label):
    ''' 
    transform a gate in quantumcircuit of quafu(not include measure_gate),
    into a node in the graph, with specific label.

    Args:
        input_gate: a gate in quantumcircuit of quafu(not include measure_gate)
        specific_label: the label of the node in the graph

    Returns:
        node: a node in the graph, with specific label. A node is a InstructionNode object.
    
    '''

    import copy
    gate = copy.deepcopy(input_gate)  # avoid modifying the original gate
    if not isinstance(gate.pos, list):  # if gate.pos is not a list, make it a list
        gate.pos = [gate.pos]

    # use getattr check 'paras' and other attributes if exist. if the attr doesn't exist,return None
    gate.paras = getattr(gate, 'paras', None) or None
    gate.duration = getattr(gate, 'duration', None) or None
    gate.unit = getattr(gate, 'unit', None) or None

    if gate.paras and not isinstance(gate.paras, list):  # if paras is True and not a list, make it a list
        gate.paras = [gate.paras]

    # hashable_gate = InstructionNode(gate.name, gate.pos, gate.paras,gate.matrix,gate.duration,gate.unit, label=i)
    hashable_gate = InstructionNode(gate.name, gate.pos, gate.paras, gate.duration, gate.unit, label=specific_label)
    return hashable_gate


# Building a DAG Graph using DAGCircuit from a QuantumCircuit
def circuit_to_dag(circuit: QuantumCircuit, measure_flag=True):
    """
    Building a DAG Graph using DAGCircuit from a QuantumCircuit

    Args:
        circuit: a QuantumCircuit object
        measure_flag: whether to add measure_gate node to the dag graph

    Returns:
        g: a DAGCircuit object

    example:
        .. jupyter-execute::

            from circuit_dag import circuit_to_dag, dag_to_circuit, draw_dag
            from quafu import QuantumCircuit

            # Create a quantum circuit as an example that you can modify as needed
            circuit = QuantumCircuit(2)
            circuit.h(0)
            circuit.cnot(0, 1)

            # Build the dag graph
            dep_graph = circuit_to_dag(circuit)  #  dag graph
    """

    # Starting Label Index
    i = 0

    # A dictionary to store the last use of any qubit
    qubit_last_use = {}

    # g = nx.MultiDiGraph()  # two nodes can have multiple edges
    # g = nx.DiGraph()   # two nodes can only have one edge
    g = DAGCircuit()  # two nodes can only have one edge

    # Add the start node 
    # g.add_node(-1,{"color": "green"})
    g.add_nodes_from([(-1, {"color": "green"})])

    # deepcopy the circuit to avoid modifying the original circuit
    # gates = copy.deepcopy(circuit.gates) # need to import copy
    # change to: gate = copy.deepcopy(input_gate) in gate_to_node()

    for gate in circuit.gates:
        # transform gate to node
        hashable_gate = gate_to_node(gate, specific_label=i)
        i += 1

        g.add_node(hashable_gate, color="blue")

        # Add edges based on qubit_last_use; update last use
        for qubit in hashable_gate.pos:
            if qubit in qubit_last_use:
                g.add_edge(qubit_last_use[qubit], hashable_gate, label=f'q{qubit}')
            else:
                g.add_edge(-1, hashable_gate, label=f'q{qubit}', color="green")

            qubit_last_use[qubit] = hashable_gate

    if measure_flag:
        # Add measure_gate node
        measure_pos = copy.deepcopy(circuit.measures)  # circuit.measures is a dict
        measure_gate = InstructionNode("measure", measure_pos, None, None, None, label="m")
        g.add_node(measure_gate, color="blue")
        # Add edges from qubit_last_use[qubit] to measure_gate
        for qubit in measure_gate.pos:
            if qubit in qubit_last_use:
                g.add_edge(qubit_last_use[qubit], measure_gate, label=f'q{qubit}')
            else:
                g.add_edge(-1, measure_gate, label=f'q{qubit}', color="green")

            qubit_last_use[qubit] = measure_gate

    # Add the end node
    # g.add_node(float('inf'),{"color": "red"})
    g.add_nodes_from([(float('inf'), {"color": "red"})])

    for qubit in qubit_last_use:
        g.add_edge(qubit_last_use[qubit], float('inf'), label=f'q{qubit}', color="red")

    # update  qubits_used, cbits_used, num_instruction_nodes
    g.update_qubits_used()
    g.update_cbits_used()
    g.update_num_instruction_nodes()
    g.update_circuit_qubits(circuit.num)

    return g


# transform gate in dag nodes  to gate in circuit which can be added to circuit
gate_classes = {
    "x": XGate,
    "y": YGate,
    "z": ZGate,
    "h": HGate,
    "s": SGate,
    "sdg": SdgGate,
    "t": TGate,
    "tdg": TdgGate,
    "rx": RXGate,
    "ry": RYGate,
    "rz": RZGate,
    "id": IdGate,
    "sx": SXGate,
    "sxdg": SXdgGate,
    "sy": SYGate,
    "sydg": SYdgGate,
    "w": WGate,
    "sw": SWGate,
    "p": PhaseGate,
    "delay": Delay,
    "barrier": Barrier,
    "cx": CXGate,
    "cp": CPGate,
    "swap": SwapGate,
    "iswap": ISwapGate,
    "rxx": RXXGate,
    "ryy": RYYGate,
    "rzz": RZZGate,
    "cy": CYGate,
    "cz": CZGate,
    "cs": CSGate,
    "ct": CTGate,
    "xy": XYResonance,
    "ccx": ToffoliGate,
    "cswap": FredkinGate,
    "mcx": MCXGate,
    "mcy": MCYGate,
    "mcz": MCZGate,
}


def node_to_gate(node_in_dag):
    """
    transform gate in dag graph, to gate in circuit which can be added to circuit

    Args:
        node_in_dag: a node in dag graph , node_in_dag is a InstructionNode object
            in GateWrapper, node_in_dag.name is uppercase, node_in_dag.pos is a list or a dict
            gate_transform support gate with one qubit or more qubits, not measures!
            and you should exculde nodes [-1 ,float('inf') , measure_gate] in dag graph

    Returns:
        gate: gate  which can be added to circuit in quafu

    example:
            import networkx as nx
            from quafu import QuantumCircuit
            qcircuit = QuantumCircuit(n)

            for gate in nx.topological_sort(dep_graph):
            
                if gate not in [-1, float('inf')]:
                    # measure gate to do
                    if gate.name == "measure":
                        qcircuit.measures = gate.pos

                    else:
                        # use gate_transform to transform gate in dag graph to gate in circuit
                        qcircuit.gates.append(node_to_gate(gate))
            return qcircuit
    
    """
    import copy
    gate_in_dag = copy.deepcopy(node_in_dag)

    gate_name = gate_in_dag.name.lower()
    gate_class = gate_classes.get(gate_name)

    if not gate_class:
        raise ValueError("gate is not supported")

    if gate_name == "barrier":
        return gate_class(gate_in_dag.pos)

    # 从gate_in_dag获取参数列表
    args = gate_in_dag.pos
    if gate_in_dag.paras:
        args += gate_in_dag.paras

    # 处理 gate.duration 和 gate.unit
    if gate_name in ["delay", "xy"]:
        args.append(gate_in_dag.duration)
        args.append(gate_in_dag.unit)

    # 处理多量子比特门
    if gate_name in ["mcx", "mcy", "mcz"]:
        control_qubits = gate_in_dag.pos[:-1]
        target_qubit = gate_in_dag.pos[-1]
        return gate_class(control_qubits, target_qubit)
    # print('gate_in_dag', gate_in_dag)
    return gate_class(*args)


# From DAG with Hashable Gates to quafu Gates added to circuit
def dag_to_circuit(dep_graph, n: int):
    '''
    From DAG with Hashable Gates to quafu Gates added to circuit
    
    Args:
        dep_graph (DAG): DAG with Hashable Gates
        n (int): number of qubits
    
    Returns:
        qcircuit (QuantumCircuit): quafu QuantumCircuit
        
    example:
        .. jupyter-execute::

            from circuit_dag import circuit_to_dag, dag_to_circuit, draw_dag
            from quafu import QuantumCircuit

            # Create a quantum circuit as an example that you can modify as needed
            circuit = QuantumCircuit(2)
            circuit.h(0)
            circuit.cnot(0, 1)

            # Build the dag graph
            dep_graph = circuit_to_dag(circuit)  #  dag graph  
            
            # use dag_to_circuit to transform dag graph to a new circuit
            reconstructed_circuit = dag_to_circuit(dep_graph, circuit.num)
        
     
    '''

    qcircuit = QuantumCircuit(n)
    # print('222222222222222dep_graph', dep_graph.edges())
    # print('is_directed_acyclic_graph', nx.is_directed_acyclic_graph(dep_graph))
    # list_nodes = list(nx.topological_sort(dep_graph))
    # print('list_nodes', list_nodes)
    # print('dagcircuit is dag?',dep_graph.is_dag())
    # show_dag(dep_graph)
    for gate in list(nx.topological_sort(dep_graph)):
        # for gate in list_nodes:
        #     print('gate',gate)

        if gate not in [-1, float('inf')]:
            # measure gate to do
            if gate.name == "measure":
                qcircuit.measures = gate.pos

            else:
                # use gate_transform to transform gate in dag graph to gate in circuit
                qcircuit.gates.append(node_to_gate(gate))
    return qcircuit


# Helper function to visualize the DAG,check the example in the docstring
def draw_dag(dep_g, output_format="png"):
    '''
    Helper function to visualize the DAG

    Args:
        dep_g (DAG): DAG with Hashable Gates
        output_format (str): output format, "png" or "svg"

    Returns:
        img (Image or SVG): show the image of DAG, which is Image(filename="dag.png") or SVG(filename="dag.svg")

    example:
        .. jupyter-execute::
        ex1:
            # directly draw  PNG picture
            draw_dag(dep_g, output_format="png")    # save a png picture "dag.png" and show it in jupyter notebook

            # directly draw  SVG   picture
            draw_dag(dep_g, output_format="svg")    # save a svg picture "dag.svg" and show it in jupyter notebook
       
        ex2:
            # generate   PNG  picture
            img_png = draw_dag(dep_g, output_format="png") 

            # generate   SVG  picture
            img_svg = draw_dag(dep_g, output_format="svg") 
            
            # show PNG picture
            img_png
            
            # show SVG picture
            img_svg        

            
    '''
    import pygraphviz

    write_dot(dep_g, "dag.dot")
    G = pygraphviz.AGraph("dag.dot")
    G.layout(prog="dot")

    if output_format == "png":
        G.draw("dag.png")
        return Image(filename="dag.png")
    elif output_format == "svg":
        G.draw("dag.svg")
        return SVG(filename="dag.svg")
    else:
        raise ValueError("Unsupported output format: choose either 'png' or 'svg'")


def nodelist_to_dag(op_nodes: List[Any]) -> DAGCircuit:
    # Starting Label Index
    i = 0

    # A dictionary to store the last use of any qubit
    qubit_last_use = {}

    # g = nx.MultiDiGraph()  # two nodes can have multiple edges
    # g = nx.DiGraph()   # two nodes can only have one edge
    g = DAGCircuit()

    # Add the start node 
    # g.add_node(-1,{"color": "green"})
    g.add_nodes_from([(-1, {"color": "green"})])

    # deepcopy the circuit to avoid modifying the original circuit
    # gates = copy.deepcopy(circuit.gates) # need to import copy
    # change to: gate = copy.deepcopy(input_gate) in gate_to_node()

    for op_node in op_nodes:
        # transform gate to node
        hashable_gate = copy.deepcopy(op_node)
        g.add_node(hashable_gate, color="blue")

        # # my add
        # if len(hashable_gate.pos) == 1 and hashable_gate.paras is None:
        #     hashable_gate.paras = 6.283185307179586

        # Add edges based on qubit_last_use; update last use
        for qubit in hashable_gate.pos:
            if qubit in qubit_last_use:
                g.add_edge(qubit_last_use[qubit], hashable_gate, label=f'q{qubit}')
            else:
                g.add_edge(-1, hashable_gate, label=f'q{qubit}', color="green")

            qubit_last_use[qubit] = hashable_gate

    # Add the end node
    # g.add_node(float('inf'),{"color": "red"})
    g.add_nodes_from([(float('inf'), {"color": "red"})])

    for qubit in qubit_last_use:
        g.add_edge(qubit_last_use[qubit], float('inf'), label=f'q{qubit}', color="red")

    # update the  qubits_used, cbits_used, num_instruction_nodes
    g.qubits_used = g.update_qubits_used()
    g.cbits_used = g.update_cbits_used()
    g.num_instruction_nodes = g.update_num_instruction_nodes()

    return g


# nodes_qubit_mapping_dict
def nodelist_qubit_mapping_dict(nodes_list):
    '''
    Args:
        nodes_list: a list of nodes
    Returns:
        nodes_qubit_mapping_dict: a dict about keys are the qubits used by the nodes and values are the new qubits
    '''
    nodes_list_qubits_used = set()
    for node in nodes_list:
        if hasattr(node, 'pos') and node.pos is not None:
            nodes_list_qubits_used = nodes_list_qubits_used | set(node.pos)

    mapping_pos = list(range(len(nodes_list_qubits_used)))
    # mapping, get a dict
    nodes_qubit_mapping_dict = dict(zip(sorted(list(nodes_list_qubits_used)), mapping_pos))

    return nodes_qubit_mapping_dict


def nodelist_qubit_mapping_dict_reverse(nodes_list):
    '''
    Args:
        nodes_list: a list of nodes
    Returns:
        nodes_qubit_mapping_dict_reverse: a dict about keys are the new qubits and values are the qubits used by the nodes
    '''
    nodes_qubit_mapping_dict = nodelist_qubit_mapping_dict(nodes_list)
    # reverse mapping, get a dict
    nodes_qubit_mapping_dict_reverse = {value: key for key, value in nodes_qubit_mapping_dict.items()}

    return nodes_qubit_mapping_dict_reverse


# a function to map nodes_list
def nodes_list_mapping(nodes_list, nodes_qubit_mapping_dict):
    '''
    Args:
        nodes_list: the nodes list of instruction nodes
        nodes_qubit_mapping_dict: the dict of the mapping qubits

    return: 
        nodes_list_mapping: the nodes_list after mapping qubits
    '''
    nodes_list_mapping = []
    for node in nodes_list:
        node_new = copy.deepcopy(node)
        if hasattr(node, 'pos') and node.pos is not None:
            if isinstance(node.pos, list):
                node_new.pos = [nodes_qubit_mapping_dict[qubit] for qubit in node.pos]
            elif isinstance(node.pos, dict):
                node_new.pos = {}
                # the values of the dict are void, so we need to copy the values from the original dict
                for qubit in node.pos:
                    node_new.pos[nodes_qubit_mapping_dict[qubit]] = copy.deepcopy(node.pos[qubit])
        nodes_list_mapping.append(node_new)
    return nodes_list_mapping


def copy_dag(dag: DAGCircuit):
    """
    Copy a DAGCircuit object.
    :param dag: DAGCircuit
    :return: DAGCircuit
    """
    dag_copy = DAGCircuit()
    dag_copy.circuit_qubits = copy.deepcopy(dag.circuit_qubits)
    dag_copy.qubits_used = copy.deepcopy(dag.qubits_used)
    dag_copy.cbits_used = copy.deepcopy(dag.cbits_used)
    dag_copy.num_instruction_nodes = copy.deepcopy(dag.num_instruction_nodes)
    dag_copy.add_weighted_edges_from(copy.deepcopy(dag.edges(data=True)))
    return dag_copy

def show_dag(dag: DAGCircuit) -> None:
    """
    Show the DAG of a circuit in a figure with proper size in interactive mode.
    This function is used for debugging such as in PyCharm and other IDEs.
    :param dag:DAGCircuit
    :return:
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    draw_dag(dag)
    im = Image.open('dag.png')

    # Convert to array and plot.
    data = np.array(im)
    # print("data.shape",data.shape)
    original_width, original_height = data.shape[1], data.shape[0]
    # Calculate the scaling ratio so that the maximum size does not exceed 2^16.
    max_size = 2 ** 16
    scale = min(max_size / original_width, max_size / original_height)

    # Calculate the new width and height.
    if scale < 10:
        new_width = int(original_width * scale) // 10
        new_height = int(original_height * scale) // 10
    else:  # Smooth drawing in small size.
        new_width = int(original_width)
        new_height = int(original_height)

    # Assuming 100 dpi for display in matplotlib.pyplot.figure
    plt.figure(figsize=(new_width // 100, new_height // 100))
    plt.imshow(data)
    plt.axis('off')
    plt.show()

