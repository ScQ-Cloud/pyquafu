from quafu import QuantumCircuit, simulate
# from quafu.elements.element_gates import *
import quafu.elements.element_gates as qeg
# from quafu.elements.parameters import Parameter
import math


# c = QuantumCircuit(2)
# c.x(0)
# c.rx(1, 0.5)
# g = c.parameterized_gates[0]
# print("\n ------ Testing ------ \n")
# c.draw_circuit()
# assert isinstance(g, RXGate)
# assert math.isclose(g.paras[0], 0.5)

#-----simulate test-----
# pq = QuantumCircuit(4)
# theta = [Parameter("theta_%d" %(i), i+1) for i in range(4)]

# for i in range(4):
#     pq.rx(i, theta[i])

# pq.ry(2, theta[0]*theta[1]-3.*theta[0])
# pq.rxx(2, 3, 0.4)
# pq.draw_circuit()

# # q = QuantumCircuit(2)
# # q.h(0)
# # q.cx(0, 1)
# res = simulate(pq, output="state_vector")

# #----controll test----
# q = QuantumCircuit(3)
# q << (qeg.XGate(1))
# q << (qeg.CXGate(0, 2))
# q << (qeg.RXGate(0, 0.1))
# q << (qeg.XGate(2)).ctrl_by([0])
# q << qeg.RZZGate(0, 2, 0.26)
# nq = q.add_controls(2)
# nq.draw_circuit()
# try:
#     nq.to_openqasm()
# except NotImplementedError as e:
#     print(e)

# c = QuantumCircuit(2)
# c.x(0)
# c.rx(1, 0.5)
# c.draw_circuit()
# c.update_params([0.1])
# g = c.parameterized_gates[0]
# c.draw_circuit()
# assert isinstance(g, RXGate)
# assert math.isclose(g.paras[0], 0.1)
# c.update_params([0.2])
# assert math.isclose(g.paras[0], 0.2)
# c.update_params([None])

# #test collapse
# qc = QuantumCircuit(2)
# qc.x(0)
# qc.measure([0], [0])
# qc.cx(0, 1)
# qc.measure([1], [1])

# result = simulate(qc, shots=10)
# probs = result.probabilities
# counts = result.count
# print(counts)
import numpy as np

qc = QuantumCircuit(2)
qc.x(0)
qc.measure([0], [0])
qc.reset([0])
qc.measure([0], [1])

psi = np.zeros(4, dtype=np.complex128)
psi[0] = 1.
result = simulate(qc, shots=10, psi = psi)
probs = result.probabilities
counts = result.count
print(counts)