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
    for i in range(num_qubits):
        pauli = "Z" + str(i)
        obs_list.append(Hamiltonian.from_pauli_list([(pauli, 1)]))
    return obs_list


# TODO(zhaoyilun): support more measurement types
# FIXME(zhaoyilun): remove backend
def run_circ(
    circ: QuantumCircuit,
    params: Optional[List[float]] = None,
    backend: str = "sim",
    estimator: Optional[Estimator] = None,
):
    """Execute a circuit

    Args:
        circ (QuantumCircuit): circ
        params (Optional[List[float]]): params
        backend (str): backend
        estimator (Optional[Estimator]): estimator
    """
    obs_list = _generate_expval_z(circ.num)
    if estimator is None:
        estimator = Estimator(circ, backend=backend)
    if params is None:
        params = [g.paras for g in circ.parameterized_gates]
    output = [estimator.run(obs, params, cache_key="00") for obs in obs_list]
    estimator.clear_cache()
    return np.array(output)


# TODO(zhaoyilun): support more gradient methods
def jacobian(
    circ: QuantumCircuit,
    params_input: np.ndarray,
    estimator: Optional[Estimator] = None,
):
    """Calculate Jacobian matrix

    Args:
        circ (QuantumCircuit): circ
        params_input (np.ndarray): params_input, with shape [batch_size, num_params]
        estimator (Estimator): estimator for calculating expectations.


    Notes:
        Since now we only use Z-axis expectations for all qubits as outputs
        i.e., the observable is Z0,Z1,..., for the same circuit we only need
        to send one task and the execution results can be used for calculating
        expectations for all these Pauli-Z operators.

        Thus we use cache here, to uniquely identity a circuit, we use the id
        of parameters. Here we have batch_size * num_parameters * 2 lists of
        parameters. Let batch_size be $M$, $N = num_parameters * 2$,

        let $i\\in\\[0, M-1\\]$, $j\\in\\[0, M-1\\]$, the cache_key is then set
        to be "{i}{j}"
    """
    batch_size, num_params = params_input.shape
    obs_list = _generate_expval_z(circ.num)
    num_outputs = len(obs_list)
    if estimator is None:
        estimator = Estimator(circ)
    calc_grad = ParamShift(estimator)
    output = np.zeros((batch_size, num_outputs, num_params))
    for i in range(batch_size):
        # Same circuit, i.e., measurement results with the same parameters may be reused
        cache_key_prefix = str(i)
        grad_list = [
            np.array(
                calc_grad(obs, params_input[i, :].tolist(), cache_key=cache_key_prefix)
            )
            for obs in obs_list
        ]
        output[i, :, :] = np.stack(grad_list)
    estimator.clear_cache()
    return output


def compute_vjp(jac: np.ndarray, dy: np.ndarray):
    r"""
    Compute vector-jacobian product.

    Args:
        jac (np.ndarray): jac with shape (batch_size, num_outputs, num_params)
        dy (np.ndarray): dy with shape (batch_size, num_outputs)

    Notes:
        Suppose there are n inputs and m outputs in current node
        Let x, y denote the inputs and outputs of current node, o denotes the final output
        Essentially, jacobian is

        .. math::
            \begin{bmatrix}
        \frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{x_n} \\
        \vdots & \ddots & \vdots \\
        \frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{x_n}
            \end{bmatrix}

        `dy` is actually the vjp of dependent node

        .. math:: \[ \frac{partial o}{partial y_1} \dots \frac{partial o}{partial y_m} \]

        Therefore the vector jocobian product gets

        .. math:: \[ \frac{partial o}{partial x_1} \dots \frac{partial o}{partial x_n} \]
    """
    batch_size, num_outputs, _ = jac.shape
    assert dy.shape[0] == batch_size and dy.shape[1] == num_outputs

    # Compute vector-Jacobian product using Einstein summation convention
    #   the scripts simply mean 'jac-dims,dy-dims->vjp-dims'; so num_outputs is summed over
    return np.einsum("ijk,ij->ik", jac, dy)
