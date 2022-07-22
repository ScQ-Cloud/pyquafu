
#%%-------
import numpy as np
from scqkit.quantum_circuit import QuantumCircuit
import copy
import time
q = QuantumCircuit(2)
q.h(0)
q.cnot(0, 1)
q.measure([0, 1], shots=1000, tomo=False)
print(q.to_openqasm())
q.draw_circuit()
q.set_backend("ScQ-P10")
res = q.send()
res.plot_amplitudes()
# simu_res = q._simulate('prob')
# print(simu_res)
# res.transpiled_circuit.draw_circuit()
# print(res.transpiled_openqasm)


#%%--------------
import numpy as np
from scqkit.quantum_circuit import QuantumCircuit
import copy
q = QuantumCircuit(5)

for i in range(5):
    if i % 2 == 0:
        q.x(i)

q.barrier([0])
q.cnot(2, 1)
q.cnot(2, 4)
q.h(0)
q.ry(1, np.pi/2)
q.rx(2, np.pi)
q.rz(3, 0.1)
q.cz(2, 3)
q.z(3)
q.y(1)
q.barrier([0, 1])
measures = [0, 1, 2, 3, 4]
cbits = [2, 0, 1, 3, 4]
q.measure(measures, 1000, cbits=cbits)
q.draw_circuit()
print(q.to_qLisp())
# res = q.send()
# res.plot_amplitudes()

#%%-------test for openqasm--------
from scqkit.quantum_circuit import QuantumCircuit
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
from scqkit.quantum_circuit import QuantumCircuit
q = QuantumCircuit(5)
measures = [0, 1, 2, 3, 4]
test_Ising = [["X", [i]] for i in range(5)]
test_Ising.extend([["ZZ", [i, i+1]] for i in range(4)])
for i in range(5):
    if i % 2 == 0:
        q.h(i)

q.set_backend("ScQ-P20")
cbits = [2, 0, 1, 4, 3]
q.measure(measures, 1000, cbits=cbits)
q.draw_circuit()
res, obsexp = q.submit_task(test_Ising)
res[0].plot_amplitudes()
E = sum(obsexp)
print(obsexp)

