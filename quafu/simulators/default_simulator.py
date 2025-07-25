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
from scipy.sparse import coo_matrix, eye, kron
from sparse import COO, SparseArray

from ..circuits import QuantumCircuit
from ..elements import Barrier, Delay, QuantumGate, XYResonance
from ..results.results import SimuResult


def global_op(gate: QuantumGate, global_qubits: List) -> coo_matrix:
    """Local operators to global operators"""
    num = len(global_qubits)
    if len(gate.pos) == 1:
        local_mat = coo_matrix(gate.matrix)
        pos = global_qubits.index(gate.pos)
        return kron(kron(eye(2**pos), local_mat), eye(2 ** (num - pos - 1)))

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
    return kron(kron(eye(2**num_left), center_mat), eye(2**num_right))


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
    return np.reshape(mat, [2**num] * r)


def ptrace(psi, ind_a: List, diag: bool = True) -> np.ndarray:
    """partial trace on a state vector"""
    num = int(np.log2(psi.shape[0]))
    order = copy.deepcopy(ind_a)
    order.extend([p for p in range(num) if p not in ind_a])

    psi = permutebits(psi, order)
    if diag:
        psi = np.abs(psi) ** 2
        psi = np.reshape(psi, [2 ** len(ind_a), 2 ** (num - len(ind_a))])
        psi = np.sum(psi, axis=1)
        return psi
    psi = np.reshape(psi, [2 ** len(ind_a), 2 ** (num - len(ind_a))])
    return psi @ np.conj(np.transpose(psi))


def py_simulate(qc: QuantumCircuit, state_ini: np.ndarray = np.array([])) -> SimuResult:
    """
    Simulate quantum circuit.

    Args:
        qc: quantum circuit need to be simulated.
        state_ini (numpy.ndarray): Input state vector

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
        if not isinstance(gate, (Barrier, Delay, XYResonance)):
            op = global_op(gate, used_qubits)
            psi = op @ psi

    return psi
