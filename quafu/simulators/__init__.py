# (C) Copyright 2023 Beijing Academy of Quantum Information Sciences
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simulator Module."""
from typing import Union

import numpy as np

from ..circuits.quantum_circuit import QuantumCircuit
from ..results.results import SimuResult


# pylint: disable=too-many-arguments, too-many-positional-arguments
def simulate(
    qc: Union[QuantumCircuit, str],
    psi: np.ndarray = np.array([]),
    simulator: str = "statevector",
    shots: int = 0,
    hamiltonian=None,
    use_gpu: bool = False,
    use_custatevec: bool = False,
) -> SimuResult:
    """
    Simulate quantum circuit.

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
        SimuResult object that contain the results.
    """

    qasm = ""
    if isinstance(qc, str):
        qasm = qc
        qc = QuantumCircuit(0)
        qc.from_openqasm(qasm)

    # simulate
    if simulator == "statevector":
        # pylint: disable=import-outside-toplevel
        from .simulator import SVSimulator

        backend = SVSimulator(use_gpu, use_custatevec)
        return backend.run(qc, psi, shots, hamiltonian)
    if simulator == "noisy statevetor":
        # pylint: disable=import-outside-toplevel
        from .simulator import NoiseSVSimulator

        backend = NoiseSVSimulator(use_gpu, use_custatevec)
        return backend.run(qc, psi, shots, hamiltonian)
    if simulator == "clifford":
        # pylint: disable=import-outside-toplevel
        from .simulator import CliffordSimulator

        return CliffordSimulator().run(qc, shots)

    raise ValueError("invalid simulator name")
