from quafu import Task, QuantumCircuit, User
from quafu.simulators.qutip_simulator import simulate 
user = User()
user.save_apitoken("eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpZCI6NTUsImV4cCI6MTY2MjQ0NzAzNX0.HUw-X_qWykae_2sc-VxSJdP4HUjuvZJJHaH762k2378")

t = Task()
t.load_account()
t.config(backend="ScQ-P10", shots=2000, compile=True)

qc = QuantumCircuit(5)
qc.h(0)
for i in range(4):
    qc.cnot(i, i+1)

meas = [0, 1, 2, 3, 4]
qc.measure(meas, cbits=[0, 1, 2, 3, 4])
qc.draw_circuit()
res = t.send(qc)
res.plot_amplitudes()

simu_res = simulate(qc)
simu_res.plot_amplitudes(full=True)


