from typing import Union
from .default_simulator import py_simulate, ptrace, permutebits
from .qfvm import simulate_circuit, execute
from quafu import QuantumCircuit
from ..results.results import SimuResult
import numpy as np
from ..exceptions import QuafuError


def simulate(qc: Union[QuantumCircuit, str],
             psi: np.ndarray = np.array([]),
             simulator: str = "qfvm_circ",
             output: str = "probabilities",
             use_gpu: bool = False,
             use_custatevec: bool = False) -> SimuResult:
    """Simulate quantum circuit
    Args:
        qc: quantum circuit or qasm string that need to be simulated.
        psi : Input state vector
        simulator:`"qfvm_circ"`: The high performance C++ circuit simulator with optional GPU support. 
                `"py_simu"`: Python implemented simulator by sparse matrix with low performace for large scale circuit.
                `"qfvm_qasm"`: The high performance C++ qasm simulator with limited gate set.

        output: `"probabilities"`: Return probabilities on measured qubits, ordered in big endian convention.
                `"density_matrix"`: Return reduced density_amtrix on measured qubits, ordered in big endian convention.
                `"state_vector`: Return original full statevector. The statevector returned by `qfvm` backend is ordered in little endian convention (same as qiskit), while `py_simu` backend is orderd in big endian convention.
        use_gpu: Use the GPU version of `qfvm_circ` simulator.
        use_custatevec: Use cuStateVec-based `qfvm_circ` simulator. The argument `use_gpu` must also be True.

    Returns:
        SimuResult object that contain the results.
"""
    qasm = ""
    if simulator == "qfvm_qasm":
        if not isinstance(qc, str):
            raise ValueError("Must input valid qasm str for qfvm_qasm simulator")

        qasm = qc
        qc = QuantumCircuit(0)
        qc.from_openqasm(qasm)

    measures = [qc.used_qubits.index(i) for i in qc.measures.keys()]
    num = 0
    if simulator == "qfvm_circ":
        num = max(qc.used_qubits) + 1
        measures = list(qc.measures.keys())
        if use_gpu:
            if use_custatevec:
                try:
                    from .qfvm import simulate_circuit_custate
                except ImportError:
                    raise QuafuError(" pyquafu is installed with cuquantum support")
                psi = simulate_circuit_custate(qc, psi)
            else:
                try:
                    from .qfvm import simulate_circuit_gpu
                except ImportError:
                    raise QuafuError("you are not using the GPU version of pyquafu")
                psi = simulate_circuit_gpu(qc, psi)
        else:
            psi = simulate_circuit(qc, psi)

    elif simulator == "py_simu":
        psi = py_simulate(qc, psi)
    elif simulator == "qfvm_qasm":
        num = qc.num
        measures = list(qc.measures.keys())
        psi = execute(qasm)
    else:
        raise ValueError("invalid circuit")

    if output == "density_matrix":
        if simulator in ["qfvm_circ", "qfvm_qasm"]:
            psi = permutebits(psi, range(num)[::-1])
        rho = ptrace(psi, measures, diag=False)
        rho = permutebits(rho, list(qc.measures.values()))
        return SimuResult(rho, output)

    elif output == "probabilities":
        if simulator in ["qfvm_circ", "qfvm_qasm"]:
            psi = permutebits(psi, range(num)[::-1])
        probabilities = ptrace(psi, measures)
        probabilities = permutebits(probabilities, list(qc.measures.values()))
        return SimuResult(probabilities, output)

    elif output == "state_vector":
        return SimuResult(psi, output)

    else:
        raise ValueError("output should in be 'density_matrix', 'probabilities', or 'state_vector'")
