import graphviz

from quafu import QuantumCircuit
from quafu.dagcircuits.circuit_dag import circuit_to_dag
from quafu.dagcircuits.instruction_node import InstructionNode


def _extract_node_info(node):
    if isinstance(node, InstructionNode):
        name, label = str(id(node)), str(node)
    else:
        assert node == -1 or node == float("inf")
        name, label = str(node), str(node)
    return name, label


def draw_dag(qc: QuantumCircuit):
    """
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


        """
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
