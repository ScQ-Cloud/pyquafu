import networkx as nx
import copy
from quafu import QuantumCircuit
from quafu.transpile.graph.circuitgraph import circuit_to_graph, relabel_graph
from quafu.transpile.graph.graphkernel import wl_subtree_kernel, fast_subtree_kernel


def similar_struct(circuit: QuantumCircuit, sub_data):
    """ Compare the similarity of weighted graph of circuit and substructure of quantum chip.

    Args:
        circuit(QuantumCircuit)
        sub_data: backend library data e.g. substructure_data['substructure_dict']
    Returns:
        most_similar_struct(list): Chip substructures similar to quantum circuits,
                                    sorted according to the value of the kernel,
                                    the higher the kernel value, the more similar.
    """
    g1 = circuit_to_graph(circuit)
    similar_struct_list = []
    kernel_value_list = []
    cut = 100  # number of substructures to search
    k = 0
    for sub_list in sub_data:
        if k < cut:
            g2 = nx.Graph()
            g1_copy = copy.deepcopy(g1)
            for item in sub_list:
                g2.add_edges_from([(item[0], item[1], {'weight': item[2]})])
            g2 = relabel_graph(g2)

            # # W-L subtree kernel iteration
            kernel_value = wl_subtree_kernel(g1_copy, g2, iteration=5)
            if round(kernel_value, 0) not in kernel_value_list:
                kernel_value_list.append(round(kernel_value, 0))
                similar_struct_list.append((sub_list, kernel_value))

            # # fast subtree kernel iteration
            # g2 = nx.convert_node_labels_to_integers(g2)
            # kernel_value = fast_subtree_kernel(g1_copy, g2, iteration=2)
            # if round(kernel_value, 0) not in kernel_value_list:
            #     kernel_value_list.append(round(kernel_value, 0))
            #     similar_struct_list.append((sub_list, kernel_value))

            g1_copy.clear()
            g2.clear()
            k += 1

    similar_struct_list = sorted(similar_struct_list, key=lambda x: x[1], reverse=True)
    return similar_struct_list
