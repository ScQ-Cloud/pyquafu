from quafu.algorithms.hamiltonian import Hamiltonian, PauliOp
from quafu.elements.element_gates import CXGate, HGate, XGate
from quafu.elements.noise import AmplitudeDamping, Depolarizing
from quafu.simulators.simulator import NoiseSVSimulator

from quafu import QuantumCircuit


class NoisySimuTest:
    def test_noise_simu(self):
        q = QuantumCircuit(2)
        q << XGate(0) << Depolarizing(0, 0.1)
        q << XGate(1) << AmplitudeDamping(1, 0.2)
        q.measure([0])
        simulator = NoiseSVSimulator()
        hamil = Hamiltonian([PauliOp("Z0"), PauliOp("Z1")])
        res = simulator.run(q, shots=3000, hamiltonian=hamil)
        print(res["counts"])
        print(res["pauli_expects"])

    def test_noise_circuit(self):
        q = QuantumCircuit(5)
        q << HGate(0)
        q << CXGate(0, 1) << CXGate(1, 2) << CXGate(2, 3) << CXGate(3, 4)
        q.add_noise("depolarizing", channel_args=(0.02,), gates=["cx"], qubits=[0, 2, 3])
        q.measure([0, 4])
        q.draw_circuit()
        simulator = NoiseSVSimulator()
        res = simulator.run(q, shots=3000)
        print(res["counts"])
        res.plot_probabilities(from_counts=True)


if __name__ == "__main__":
    NoisySimuTest().test_noise_simu()
