from ..circuits.quantum_circuit import QuantumCircuit
from typing import Union

import numpy as np
from quafu import QuantumCircuit
from ..results.results import SimuResult

def simulate(
    qc: Union[QuantumCircuit, str],
    psi: np.ndarray = np.array([]),
    simulator: str = "statevector",
    shots: int = 100,
    hamiltonian = None,
    use_gpu: bool = False,
    use_custatevec: bool = False,
) -> SimuResult:
    """Simulate quantum circuit
    Args:
        qc: quantum circuit or qasm string that need to be simulated.
        psi : Input state vector
        simulator:
            `"statevector"`: The high performance C++ circuit simulator with optional GPU support.
            `"clifford"`: The high performance C++ cifford circuit simulator.
            `"noisy statevetor"`: Nosiy circuit simulator implemented with statevector simulator.

        shots: The shots of simulator executions.
        use_gpu: Use the GPU version of `statevector` simulator.
        use_custatevec: Use cuStateVec-based `statevector` simulator. The argument `use_gpu` must also be True.

    Returns:
        SimuResult object that contain the results."""
    
    qasm = ""
    if isinstance(qc, str):
        qasm = qc
        qc = QuantumCircuit(0)
        qc.from_openqasm(qasm)

    # simulate
    if simulator == "statevector":
        from .simulator import SVSimulator
        backend = SVSimulator(use_gpu, use_custatevec)
        return backend.run(qc, psi, shots, hamiltonian)
    elif simulator == "noisy statevetor":
        from .simulator import NoiseSVSimulator
        backend = NoiseSVSimulator(use_gpu, use_custatevec)
        return backend.run(qc, psi, shots, hamiltonian)
    elif simulator == "clifford":
        from .simulator import CliffordSimulator
        return CliffordSimulator().run(qc, shots)

    else:
        raise ValueError("invalid simulator name")
