
#%%--------------
import numpy as np
from quantum_tools import *
from paulis import *
from quantum_circuit import QuantumCircuit
import copy



#%%---------test for generation of qasm and draw circuit----
import pickle
import numpy as np
q = QuantumCircuit(5)
measures = [0, 1, 2, 3, 4]
test_Ising = [["X", [i]] for i in range(5)]
test_Ising.extend([["ZZ", [i, i+1]] for i in range(4)])
for i in range(5):
    if i % 2 == 0:
        q.x(i)

q.barrier([0])
q.cnot(1, 2)
q.cnot(2, 4)
q.h(0)
q.ry(1, np.pi/2)
q.rx(2, np.pi)
q.rz(3, 0.1)
q.cz(2, 3)
q.z(3)
q.y(1)
q.barrier([0, 1])
q.iswap(0, 1)
q.measure(measures, 1000)
q.draw_circuit()
q.set_backend("IOP")
q.send(compiler="default")
print(q.qasm)

# #%%-----class computer simulation test for merge measure--------
# a = ["X", [1]]
# b = ["X", [2]]
# c = ["ZXY", [2, 4, 3]]
# d = ["XX", [2, 3]]
# e = ["ZZ", [1, 2]]
# f = ["YX", [2, 4]]
# g = ["YY", [1, 2]]
# test_list = [a, b, c, d, e, f, g]

# test_Ising = [["X", [i]] for i in range(5)]
# test_Ising.extend([["ZZ", [i, i+1]] for i in range(4)])

# measures = [1, 2, 3, 4]
# psi = np.random.rand(2**(len(measures))) + 1j * np.random.rand(2**(len(measures)))
# psi = psi/np.linalg.norm(psi)
# q = QuantumCircuit(5)
# q.cnot(1, 2)
# q.rx(1, 0.2)
# q.measure(measures, 1000)

# _ , measure_res = q.submit_task(psi, test_list)
# res_direct = q.direct_measure(psi, test_list)

# print("measure simu: ", measure_res)
# print("direct_calc: ", res_direct) 

# measures = [0, 1, 2, 3, 4]
# psi = np.random.rand(2**(len(measures))) + 1j * np.random.rand(2**(len(measures)))
# psi = psi/np.linalg.norm(psi)
# q.measure(measures, 1000)
# _ , measure_res = q.submit_task(psi, test_Ising)
# res_direct = q.direct_measure(psi, test_Ising)

# print("measure simu: ", measure_res)
# print("direct_calc: ", res_direct)