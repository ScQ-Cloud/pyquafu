
from typing import Union
from .default_simulator import py_simulate, ptrace, permutebits
from .qfvm import simulate_circuit, execute
from quafu import QuantumCircuit
from ..results.results import SimuResult
import numpy as np
import time

def simulate(qc : Union[QuantumCircuit, str], psi : np.ndarray= np.array([]), simulator:str="qfvm_circ", output: str="amplitudes")-> SimuResult:
    """Simulate quantum circuit
    Args:
        qc: quantum circuit or qasm string that need to be simulated.
        psi : Input state vector
        simulator:`"qfvm_circ"`: The high performance C++ circuit simulator. 
                `"py_simu"`: Python implemented simulator by sparse matrix with low performace for large scale circuit.
                `"qfvm_qasm"`: The high performance C++ qasm simulator with limited gate set.

        output: `"amplitudes"`: Return ampliteds on measured qubits, ordered in big endian convention.
                `"density_matrix"`: Return reduced density_amtrix on measured qubits, ordered in big endian convention.
                `"state_vector`: Return original full statevector. The statevector returned by `qfvm` backend is ordered in little endian convention (same as qiskit), while `py_simu` backend is orderd in big endian convention.
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
        num = max(qc.used_qubits)+1
        measures = list(qc.measures.keys())
        psi = simulate_circuit(qc, psi)
        
    elif simulator ==  "py_simu":
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

    elif output == "amplitudes":
        if simulator in ["qfvm_circ", "qfvm_qasm"]:
            psi = permutebits(psi, range(num)[::-1])
        amplitudes = ptrace(psi, measures)
        amplitudes = permutebits(amplitudes, list(qc.measures.values()))
        return SimuResult(amplitudes, output) 
    
    elif output == "state_vector":
        return SimuResult(psi, output)

    else:
        raise ValueError("output should in be 'density_matrix', 'amplitudes', or 'state_vector'")