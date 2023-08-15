import logging

from quafu import QuantumCircuit, simulate, Task, User
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - Line %(lineno)d - %(message)s')

logging.info('Quick test started.')

qc = QuantumCircuit(num=4)

# apply (a Hadamard qu_gate on the first qubit)
qc.h(pos=0)

# measure (on the first qubit)
# qc.measure(pos=[0])
qc.measure()
qc_draw = qc.draw_circuit(return_str=True)

logging.info('built qc:\n' + qc_draw)

simu_res = simulate(qc, output='probabilities').probabilities
logging.info('simulated results:\nprobabilities = ' + str(simu_res))


user = User()
user.get_available_backends()

task = Task()
task.retrieve('30BEE0A009ADD81C')

# shots_num = 200
# backend = "ScQ-P18"  # ScQ-P10, ScQ-P18, ScQ-P136
# task.config(backend=backend, shots=shots_num, compile=True)
#
# exp_results = task.send(qc, wait=True)
# print(exp_results.probabilities)
