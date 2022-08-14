#default circuit simulator for state vector
from ..results.results import SimuResult
from ..elements.quantum_element import Barrier, SingleQubitGate, TwoQubitGate, MultiQubitGate
import numpy as np
from functools import reduce
from sparse import COO, kron, eye
import copy

def global_op(gate, global_qubits):
    num  = len(global_qubits)
    if isinstance(gate, SingleQubitGate):
        local_mat = COO(gate.matrix)
        pos = global_qubits.index(gate.pos)
        op = reduce(kron, [eye(2**pos), local_mat, eye(2**(num - pos-1))])
        return op

    elif isinstance(gate, TwoQubitGate) or isinstance(gate, MultiQubitGate):
        local_mat = COO(gate.matrix)
        pos = [global_qubits.index(p) for p in gate.pos]
        # print(pos)
        num_left = min(pos)
        num_right = num - max(pos) - 1
        num_center = max(pos) - min(pos) + 1
        # print(num_left)
        # print(num_right)
        # print(num_center)
        center_mat = kron(local_mat, eye(2**(num_center - len(pos))))
        origin_order = sorted(pos)
        origin_order.extend([p for p in range(min(pos), max(pos)+1) if p not in pos])
        new_order = np.argsort(origin_order)
        center_mat = permutebits(center_mat, new_order, 2)
        op = reduce(kron, [eye(2**num_left), center_mat, eye(2**num_right)])
        return op


def permutebits(mat, order, r=1):
    num = len(order)
    order = np.array(order)
    mat = np.reshape(mat, [2]*r*num)
    if r == 2:
        order = np.concatenate([order, order + num])
    mat = np.transpose(mat, order)
    mat = np.reshape(mat, [2**num]*r)
    return mat

def ptrace(psi, ind_A, diag=True):
    num = int(np.log2(psi.shape[0]))
    order = copy.deepcopy(ind_A)
    order.extend([p for p in range(num) if p not in ind_A])

    psi = permutebits(psi, order)
    if diag:
        psi = np.abs(psi)**2
        psi = np.reshape(psi, [2**len(ind_A), 2**(num-len(ind_A))])
        psi = np.sum(psi, axis=1)
        return psi        
    else:
        psi = np.reshape(psi, [2**len(ind_A), 2**(num-len(ind_A))]) 
        rho = psi @ np.transpose(psi)
        return rho

def simulate(qc, state_ini: np.ndarray = None, density_matrix=False):
    """Simulate quantum circuit on classical computer
        Args:
            stat_ini: Input state vector
    """

    used_qubits = qc.get_used_qubits()
    num = len(used_qubits)
    assert num <= 12
    measures = [used_qubits.index(i) for i in qc.measures.keys()]
    if state_ini is None:
        psi = np.zeros(2**num)
        psi[0] = 1

    else:
        psi = state_ini

    for gate in qc.gates:   
        if not isinstance(gate, Barrier): 
            op = global_op(gate, used_qubits)
            psi = op @ psi

    # inds = np.argsort(np.argsort(measures)
    # )
    if density_matrix:
        rho = ptrace(psi, measures, diag=False)
        rho = permutebits(rho, list(qc.measures.values()), r=2)
        return SimuResult(rho, density_matrix)
    else:
        amplitudes = ptrace(psi, measures)
        amplitudes = permutebits(amplitudes, list(qc.measures.values()))
        return SimuResult(amplitudes, density_matrix) 