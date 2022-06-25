from Qcover.core import Qcover
from Qcover.optimizers import COBYLA
from Qcover.backends import CircuitByQiskit, CircuitByCirq, CircuitByProjectq, CircuitByTensor
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import time
from collections import defaultdict
import copy
# from Qcovercompiler import QcoverCompiler
from para_circuit import QAOACircuit

physical_qubits = 3
logical_qubits = 3

# # linear hardware coupling
# linear_hardware_coupling = []
# for i in range(0, physical_qubits - 1):
#     linear_hardware_coupling.append([i, i + 1])
# hardware = nx.Graph()
# hardware.add_edges_from(linear_hardware_coupling)
# # nx.draw_networkx(hardware)
# # plt.show()

# generate random weight graph
node_num, edge_num = logical_qubits, 10
nodes, edges = Qcover.generate_graph_data(node_num, edge_num)

# graph_edges = list(g.edges)
# print(graph_edges)
# edges = {(0, 1, 2), (1, 2, 1), (2, 3, 4), (3, 4, 3), (4, 5, 5), (3, 1, 3), (4, 2, 7)}
# nodes = {(0, 4), (1, 7), (2, 9), (3, 1), (4, 2), (5, 3)}
# phys_order = [2, 0, 4, 1, 5, 3]
# edges = {(0, 1, 2), (1, 2, 5), (2, 3, 3), (0, 2, 3), (3, 4, 5)}
# nodes = {(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)}
# edges = {(2, 8, 0), (4, 3, 1), (5, 8, 1), (7, 0, 9), (8, 0, 9), (8, 1, 0), (8, 4, 3)}
# nodes = {(0, 6), (1, 3), (2, 5), (3, 0), (4, 7), (5, 5), (6, 3), (7, 4), (8, 9)}
# random_qubit = random.sample(range(0, physical_qubits-logical_qubits+1),1)
# # phys_order = list(range(random_qubit[0], random_qubit[0] + logical_qubits))
# phys_order = random.sample(range(random_qubit[0], random_qubit[0] + logical_qubits),logical_qubits)

p = 1
# # Run qcover to generate optimal parameters
# g = Qcover.generate_weighted_graph(nodes, edges)
# # qulacs_bc = CircuitByQiskit()
# # qulacs_bc = CircuitByProjectq()
# qulacs_bc = CircuitByCirq()
# # qulacs_bc = CircuitByQulacs()
# # qulacs_bc = CircuitByTensor()
# optc = COBYLA(options={'tol': 1e-3, 'disp': False})
# qc = Qcover(g, p=p, optimizer=optc, backend=qulacs_bc)
# res = qc.run()
# optimal_params = res['Optimal parameter value']
# # optimal_params = np.ones(2 * p)
# print('optimal_params:', optimal_params)


# # draw weighted graph
# new_labels = dict(map(lambda x: ((x[0], x[1]), str(x[2]['weight'])), g.edges(data=True)))
# pos = nx.spring_layout(g)
# nx.draw_networkx(g, pos=pos)
# nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=new_labels)
# nx.draw_networkx_edges(g, pos, width=2, edge_color='g', arrows=False)
# plt.show()
params = [0.1, 0.2]
gate='CNOT'
# gate='iSWAP'
qaoa = QAOACircuit(logical_qubits, physical_qubits, nodes, edges, params, p, gate=gate)
qaoa.set_backend("IOP")
qaoa.compile_to_IOP()
# print(qaoa.qasm, "\n")
qaoa.circuit_from_qasm(qaoa.qasm)
qaoa.draw_circuit()
# qaoa.send()
