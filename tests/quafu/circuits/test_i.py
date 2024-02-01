from quafu import QuantumCircuit, simulate
from quafu.elements.element_gates import *
from quafu.elements.parameters import Parameter
import math


# c = QuantumCircuit(2)
# c.x(0)
# c.rx(1, 0.5)
# g = c.parameterized_gates[0]
# print("\n ------ Testing ------ \n")
# c.draw_circuit()
# assert isinstance(g, RXGate)
# assert math.isclose(g.paras[0], 0.5)

pq = QuantumCircuit(4)
theta = [Parameter("theta_%d" %(i), i+1) for i in range(4)]

for i in range(4):
    pq.rx(i, theta[i])

pq.ry(2, theta[0]*theta[1]-3.*theta[0])
pq.rxx(2, 3, 0.4)
pq.draw_circuit()

# q = QuantumCircuit(2)
# q.h(0)
# q.cx(0, 1)
res = simulate(pq, output="state_vector")