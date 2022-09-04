#default circuit simulator for state vector
from typing import Iterable, Union
from ..results.results import SimuResult
from ..elements.quantum_element import Barrier, Delay, SingleQubitGate, TwoQubitGate, MultiQubitGate, XYResonance
import numpy as np
from functools import reduce
from sparse import COO, SparseArray
from scipy.sparse import kron, eye, coo_matrix

import copy

def global_op(gate, global_qubits):
    """Local operators to global operators"""
    num  = len(global_qubits)
    if isinstance(gate, SingleQubitGate):
        local_mat = coo_matrix(gate.matrix)     
        pos = global_qubits.index(gate.pos)
        local_mat = kron(kron(eye(2**pos), local_mat), eye(2**(num - pos-1)))
        return local_mat

    elif isinstance(gate, TwoQubitGate) or isinstance(gate, MultiQubitGate):
        local_mat =coo_matrix(gate.matrix)
        pos = [global_qubits.index(p) for p in gate.pos]
        num_left = min(pos)
        num_right = num - max(pos) - 1
        num_center = max(pos) - min(pos) + 1
        center_mat = kron(local_mat, eye(2**(num_center - len(pos))))
        origin_order = sorted(pos)
        origin_order.extend([p for p in range(min(pos), max(pos)+1) if p not in pos])
        new_order = np.argsort(origin_order)
        center_mat = COO.from_scipy_sparse(center_mat)
        center_mat = permutebits(center_mat, new_order).to_scipy_sparse()
        center_mat = kron(kron(eye(2**num_left), center_mat), eye(2**num_right))
        return center_mat


def permutebits(mat: Union[SparseArray, np.ndarray], order : Iterable):
    """permute qubits for operators or states"""
    num = len(order)
    order = np.array(order)
    r = len(mat.shape)
    mat = np.reshape(mat, [2]*r*num)
    order = np.concatenate([order+len(order)*i for i  in range(r)]) 
    mat = np.transpose(mat, order)
    mat = np.reshape(mat, [2**num]*r)
    return mat

def ptrace(psi, ind_A, diag=True):
    """partial trace on a state vector"""
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

def simulate(qc, state_ini: np.ndarray = None, output="amplitudes"):
    """Simulate quantum circuit
        Args:
            state_ini (numpy.ndarray): Input state vector
            output (str): `"amplitudes"`: Return ampliteds on measured qubits.
                          `"density_matrix"`: Return reduced density_amtrix on measured qubits.
                          `"state_vector`: Return full statevector.
        Returns:
            (SimuResult): SimuResult class that contain the results.
    """

    used_qubits = qc.get_used_qubits()
    num = len(used_qubits)
    measures = [used_qubits.index(i) for i in qc.measures.keys()]
    if state_ini is None:
        psi = np.zeros(2**num)
        psi[0] = 1

    else:
        psi = state_ini

    for gate in qc.gates:   
        if not ((isinstance(gate, Delay)) or (isinstance(gate, Barrier)) or isinstance(gate, XYResonance)): 
            op = global_op(gate, used_qubits)
            psi = op @ psi

    if output == "density_matrix":
        rho = ptrace(psi, measures, diag=False)
        rho = permutebits(rho, list(qc.measures.values()))
        return SimuResult(rho, output)

    elif output == "amplitudes":
        amplitudes = ptrace(psi, measures)
        amplitudes = permutebits(amplitudes, list(qc.measures.values()))
        return SimuResult(amplitudes, output) 
    
    elif output == "state_vector":
        return SimuResult(psi, output)

    else:
        raise ValueError("output should in be 'density_matrix', 'amplitudes', or 'state_vector'")