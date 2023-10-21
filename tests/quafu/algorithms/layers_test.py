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
import numpy as np

from quafu.algorithms.layers import jacobian, compute_vjp
from quafu.circuits.quantum_circuit import QuantumCircuit


def test_compute_vjp():
    circ = QuantumCircuit(2)
    circ.x(0)
    circ.rx(0, 0.1)
    circ.ry(1, 0.5)
    circ.ry(0, 0.1)

    params_input = np.random.randn(4, 3)
    jac = jacobian(circ, params_input)

    dy = np.random.randn(4, 2)
    vjp = compute_vjp(jac, dy)

    assert len(vjp.shape) == 1
    assert vjp.shape[0] == 3
