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
"""default circuit simulator for state vector"""

import copy
from typing import Iterable, List, Union

import numpy as np
from quafu.circuits.quantum_circuit import QuantumCircuit
from scipy.sparse import coo_matrix, eye, kron
from sparse import COO, SparseArray

from ..elements import (
    Barrier,
    Delay,
    QuantumGate,
    XYResonance,
)
from ..results.results import SimuResult


def global_op(gate: QuantumGate, global_qubits: List) -> coo_matrix:
    """Local operators to global operators"""
    num = len(global_qubits)
    if len(gate.pos) == 1:
        local_mat = coo_matrix(gate.matrix)
        pos = global_qubits.index(gate.pos)
        local_mat = kron(kron(eye(2**pos), local_mat), eye(2 ** (num - pos - 1)))
        return local_mat

    else:
        local_mat = coo_matrix(gate.matrix)
        pos = [global_qubits.index(p) for p in gate.pos]
        num_left = min(pos)
        num_right = num - max(pos) - 1
        num_center = max(pos) - min(pos) + 1
        center_mat = kron(local_mat, eye(2 ** (num_center - len(pos))))
        origin_order = sorted(pos)
        origin_order.extend([p for p in range(min(pos), max(pos) + 1) if p not in pos])
        new_order = np.argsort(origin_order)
        center_mat = COO.from_scipy_sparse(center_mat)
        center_mat = permutebits(center_mat, new_order).to_scipy_sparse()
        center_mat = kron(kron(eye(2**num_left), center_mat), eye(2**num_right))
        return center_mat


def permutebits(
    mat: Union[SparseArray, np.ndarray], order: Iterable
) -> Union[SparseArray, np.ndarray]:
    """permute qubits for operators or states"""
    num = len(order)
    order = np.array(order)
    r = len(mat.shape)
    mat = np.reshape(mat, [2] * r * num)
    order = np.concatenate([order + len(order) * i for i in range(r)])
    mat = np.transpose(mat, order)
    mat = np.reshape(mat, [2**num] * r)
    return mat


def ptrace(psi, ind_A: List, diag: bool = True) -> np.ndarray:
    """partial trace on a state vector"""
    num = int(np.log2(psi.shape[0]))
    order = copy.deepcopy(ind_A)
    order.extend([p for p in range(num) if p not in ind_A])

    psi = permutebits(psi, order)
    if diag:
        psi = np.abs(psi) ** 2
        psi = np.reshape(psi, [2 ** len(ind_A), 2 ** (num - len(ind_A))])
        psi = np.sum(psi, axis=1)
        return psi
    else:
        psi = np.reshape(psi, [2 ** len(ind_A), 2 ** (num - len(ind_A))])
        rho = psi @ np.conj(np.transpose(psi))
        return rho


def py_simulate(
    qc: QuantumCircuit, state_ini: np.ndarray = np.array([]), output: str = "amplitudes"
) -> SimuResult:
    """Simulate quantum circuit
    Args:
        qc: quantum circuit need to be simulated.
        state_ini (numpy.ndarray): Input state vector
        output (str): `"amplitudes"`: Return ampliteds on measured qubits.
                      `"density_matrix"`: Return reduced density_amtrix on measured qubits.
                      `"state_vector`: Return full statevector.
    Returns:
       SimuResult object that contain the results.
    """

    used_qubits = qc.used_qubits
    num = len(used_qubits)
    if not state_ini:
        psi = np.zeros(2**num)
        psi[0] = 1

    else:
        psi = state_ini

    for gate in qc.gates:
        if not (
            (isinstance(gate, Delay))
            or (isinstance(gate, Barrier))
            or isinstance(gate, XYResonance)
        ):
            op = global_op(gate, used_qubits)
            psi = op @ psi

    return psi
