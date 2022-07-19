
#%%-------
import numpy as np
from quantum_circuit import QuantumCircuit
import copy
import time

q = QuantumCircuit(18)
q.h(16)
q.cnot(16, 17)
q.measure([16, 17], shots=1000, cbits=[], tomo=False)
print(q.to_openqasm())
q.draw_circuit()
q.set_backend("ScQ-P20")
res = q.send()
res.plot_amplitudes()
# simu_res = q._simulate('prob')
# print(simu_res)

#%%--------------
import numpy as np
from quantum_circuit import QuantumCircuit
import copy
q = QuantumCircuit(5)

for i in range(5):
    if i % 2 == 0:
        q.x(i)

q.barrier([0])
q.cnot(2, 1)
# q.cnot(2, 4)
q.h(0)
q.ry(1, np.pi/2)
q.rx(2, np.pi)
q.rz(3, 0.1)
q.cz(2, 3)
q.z(3)
q.y(1)
q.barrier([0, 1])
measures = [0, 1, 2, 3, 4]
q.measure(measures, 1000)
q.draw_circuit()
res = q.send()
res.plot_amplitudes()

#%%-------test for openqasm--------
from quantum_circuit import QuantumCircuit
test_ghz = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[0];
cx q[0],q[2];
"""

q = QuantumCircuit(3)
q.from_openqasm(test_ghz)
q.draw_circuit()
res = q.send()
res.plot_amplitudes()

#%%----------test for submit_task----------
import numpy as np
from quantum_circuit import QuantumCircuit
q = QuantumCircuit(5)
measures = [0, 1, 2, 3, 4]
test_Ising = [["X", [i]] for i in range(5)]
test_Ising.extend([["ZZ", [i, i+1]] for i in range(4)])
for i in range(5):
    if i % 2 == 0:
        q.h(i)

q.measure(measures, 1000)
q.draw_circuit()
res, obsexp = q.submit_task(test_Ising)
res[0].plot_amplitudes()
print(res[0].raw_res)
E = sum(obsexp)
print(obsexp)


# #%%-----class computer simulation test for merge measure--------
# from quantum_tools import *
# from paulis import *
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