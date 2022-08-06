
import qutip
import numpy as np
from ..results.results import SimuResult

def simulate(qc, state_ini: np.ndarray = None):
    """Simulate quantum circuit on classical computer
        Args:
            stat_ini: Input state vector
    """

    used_qubits = qc.get_used_qubits()
    num = len(used_qubits)
    assert num <= 12
    measures = [used_qubits.index(i) for i in qc.measures.keys()]
    if state_ini is None:
        psi = qutip.basis([2] * num, [0] * num)
    else:
        psi = qutip.Qobj(state_ini, dims=[[2] * num, [1] * num])

    for gate in qc.gates:
        oper = gate._operator(used_qubits)
        psi = oper * psi
    
    inds = np.argsort(np.argsort(measures))
    rho = qutip.ptrace(psi, measures)
    rho = rho.permute(inds).permute(list(qc.measures.values()))
    return SimuResult(rho)
