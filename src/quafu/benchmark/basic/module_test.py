import logging

from quafu import QuantumCircuit, simulate, Task

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def build_a_circuit():
    # initialize (with 1 qubit)
    qc = QuantumCircuit(num=1)

    # apply (a Hadamard qu_gate on the first qubit)
    qc.h(pos=0)

    # measure (on the first qubit)
    qc.measure(pos=[0])
    return qc


def save_a_circuit(qc: QuantumCircuit):
    print(qc.to_openqasm())
    # with open('openqasm.txt', 'w') as f:
    #     f.write(qc.to_openqasm())
    return None


def load_a_circuit():
    f = open('openqasm.txt', 'r')
    openqasm_lines = f.read()

    qc = QuantumCircuit(num=1)
    qc.from_openqasm(openqasm_lines)
    qc.draw_circuit()
    return qc


def simulation(qc: QuantumCircuit):
    simu_res = simulate(qc, output='probabilities')
    print(simu_res.probabilities)
    return simu_res


def execution(qc: QuantumCircuit):
    task = Task()
    shots_num = 200
    backend = "ScQ-P10"  # ScQ-P10, ScQ-P18, ScQ-P136
    task.config(backend=backend, shots=shots_num, compile=True)

    exp_results = task.send(qc, wait=True)
    print(exp_results.probabilities)
    # exp_results.transpiled_circuit.plot_circuit(show=True)
    return exp_results


if __name__ == '__main__':
    qc_ = build_a_circuit()
    # simulation(qc_)
    execution(qc_)

# simu_res.plot_probabilities()
# plt.show()
