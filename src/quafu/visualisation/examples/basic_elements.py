import matplotlib.pyplot as plt
from quafu.visualisation.circuitPlot import CircuitPlotManager
from quafu.circuits.quantum_circuit import QuantumCircuit
import os

if not os.path.exists('./figures/'):
    os.mkdir('./figures/')

n = 8
qc_ = QuantumCircuit(n)
qc_.h(0)
qc_.barrier([0, 3])
qc_.x(0)
qc_.swap(0, 4)
qc_.cnot(3, 6)
qc_.rz(4, 3.2)

for k in range(10):
    qc_.x(7)
for k in range(n):
    qc_.cnot(k, k + 1)
qc_.measure([0, 1, 2, 3], [0, 1, 2, 3])

# for i in range(30):
#     qc.x(4)

cmp = CircuitPlotManager(qc_)
cmp(title='Basic Elements in Circuit')

plt.savefig('./figures/basics.png', dpi=300, transparent=True)
plt.close()
