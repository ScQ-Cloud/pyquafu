from quafu import QuantumCircuit, Task, simulate


qc = QuantumCircuit(1)

qc.h(0)

qc.measure()

task = Task()
res = task.retrieve('30BEE0A009ADD81C')
print(res.probabilities)
print(res.counts)
