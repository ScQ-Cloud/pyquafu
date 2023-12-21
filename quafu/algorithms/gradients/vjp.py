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

from typing import List, Optional

import numpy as np
from quafu.algorithms.estimator import Estimator
from quafu.algorithms.gradients import ParamShift
from quafu.algorithms.hamiltonian import Hamiltonian

from quafu import QuantumCircuit


def _generate_expval_z(num_qubits: int):
    obs_list = []
    base_pauli = "I" * num_qubits
    for i in range(num_qubits):
        pauli = base_pauli[:i] + "Z" + base_pauli[i + 1 :]
        obs_list.append(Hamiltonian.from_pauli_list([(pauli, 1)]))
    return obs_list


# TODO(zhaoyilun): support more measurement types
def run_circ(circ: QuantumCircuit, params: Optional[List[float]] = None):
    """Execute a circuit

    Args:
        circ (QuantumCircuit): circ
        params (Optional[List[float]]): params
    """
    obs_list = _generate_expval_z(circ.num)
    estimator = Estimator(circ)
    if params is None:
        params = [g.paras for g in circ.parameterized_gates]
    output = [estimator.run(obs, params) for obs in obs_list]
    return np.array(output)


# TODO(zhaoyilun): support more gradient methods
def jacobian(circ: QuantumCircuit, params_input: np.ndarray):
    """Calculate Jacobian matrix

    Args:
        circ (QuantumCircuit): circ
        params_input (np.ndarray): params_input, with shape [batch_size, num_params]
    """
    batch_size, num_params = params_input.shape
    obs_list = _generate_expval_z(circ.num)
    num_outputs = len(obs_list)
    estimator = Estimator(circ)
    calc_grad = ParamShift(estimator)
    output = np.zeros((batch_size, num_outputs, num_params))
    for i in range(batch_size):
        grad_list = [
            np.array(calc_grad(obs, params_input[i, :].tolist())) for obs in obs_list
        ]
        output[i, :, :] = np.stack(grad_list)
    return output


def compute_vjp(jac: np.ndarray, dy: np.ndarray):
    """compute vector-jacobian product

    Args:
        jac (np.ndarray): jac with shape (batch_size, num_outputs, num_params)
        dy (np.ndarray): dy with shape (batch_size, num_outputs)
    """
    batch_size, num_outputs, num_params = jac.shape
    assert dy.shape[0] == batch_size and dy.shape[1] == num_outputs

    vjp = np.zeros((batch_size, num_params))

    for i in range(batch_size):
        vjp[i] = dy[i, :].T @ jac[i, :, :]

    return vjp


# class QNode:
#     """Quantum node which essentially wraps the execution of a quantum circuit"""
#
#     def __init__(self, circ: QuantumCircuit) -> None:
#         self._circ = circ
#
#     def __call__(self):
#         return execu
