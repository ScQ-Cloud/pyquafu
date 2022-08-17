
import qutip
import numpy as np
from ..elements.quantum_element import SingleQubitGate, TwoQubitGate
from ..results.results import SimuResult
from typing import Union, Callable, List, Tuple, Iterable, Any, Optional
import functools


def _oper1q_local_to_global(_gate: SingleQubitGate, used_qubits: Optional[List[int]] = None):
    """
    Single-qubit operator from local to global

    Parameters
    ----------
    _gate: SingleQubitGate
    used_qubits: list
        The total qubits used.
    Returns
    -------
    oper: qutip.Qobj
    """

    pos = _gate.pos
    if used_qubits is None:
        used_qubits = [pos]

    num = len(used_qubits)
    assert num >= 1 and pos in used_qubits

    q_idx = used_qubits.index(pos)
    gate = qutip.Qobj(_gate.matrix, dims=[[2], [2]])
    ops = []
    for idx in range(num):
        if idx != q_idx:
            ops.append(qutip.qeye(2))
        else:
            ops.append(gate)
    oper = functools.reduce(qutip.tensor, ops)
    return oper


def _oper2q_local_to_global(_gate: TwoQubitGate, used_qubits: Optional[List[int]] = None):
    """
    Two-qubit operator from local to global

    Parameters
    ----------
    _gate: TwoQubitGate
    used_qubits: list
        The total qubits used.

    Returns
    -------
    oper: qutip.Qobj
    """
    pos = np.sort(_gate.pos).tolist()
    if used_qubits is None:
        used_qubits = pos
    num = len(used_qubits)
    assert num >= 2 and all(pos_i in used_qubits for pos_i in pos)
    pos0, pos1 = used_qubits.index(pos[0]), used_qubits.index(pos[1])
    gate = qutip.Qobj(_gate.matrix, dims=[[2, 2], [2, 2]])

    pos_diff_abs = abs(pos1 - pos0)
    if pos_diff_abs == 1:
        # two nearest neighbor qubits
        if pos0 == 0:
            if pos1 == int(num - 1):
                oper = gate
            else:
                qeye_right = qutip.qeye([2] * int(num - 1 - pos1))
                oper = qutip.tensor(gate, qeye_right)
        else:
            qeye_left = qutip.qeye([2] * int(pos0))
            if pos1 == int(num - 1):
                oper = qutip.tensor(qeye_left, gate)
            else:
                qeye_right = qutip.qeye([2] * int(num - 1 - pos1))
                oper = qutip.tensor(qeye_left, gate, qeye_right)

    else:
        # two non-nearest neighbor qubits
        oper = qutip.tensor(gate, qutip.qeye([2] * int(pos_diff_abs - 1)))
        idxs = list(range(pos_diff_abs + 1))
        idxs[1], idxs[-1] = idxs[-1], idxs[1]
        oper = oper.permute(idxs)

        if pos0 == 0:
            if pos1 < int(num - 1):
                qeye_right = qutip.qeye([2] * int(num - 1 - pos1))
                oper = qutip.tensor(oper, qeye_right)
        else:
            qeye_left = qutip.qeye([2] * int(pos0))
            if pos1 == int(num - 1):
                oper = qutip.tensor(qeye_left, oper)
            else:
                qeye_right = qutip.qeye([2] * int(num - 1 - pos1))
                oper = qutip.tensor(qeye_left, gate, qeye_right)

    return oper


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
        if isinstance(SingleQubitGate):
            oper = _oper1q_local_to_global(gate, used_qubits)
        elif isinstance(TwoQubitGate):
            oper = _oper2q_local_to_global(gate, used_qubits)

        psi = oper * psi
    
    inds = np.argsort(np.argsort(measures))
    rho = qutip.ptrace(psi, measures)
    rho = rho.permute(inds).permute(list(qc.measures.values()))
    return rho
