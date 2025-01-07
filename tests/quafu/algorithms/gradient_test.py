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

import sys

import numpy as np
import pytest
from quafu.algorithms.estimator import Estimator
from quafu.algorithms.gradients import ParamShift, grad_para_shift
from quafu.algorithms.hamiltonian import Hamiltonian
from quafu.circuits.quantum_circuit import QuantumCircuit
from quafu.elements import Parameter


# TODO: remove this test after releasing 0.4.1 as it is not necessary
class TestParamShift:
    # @pytest.mark.skipif(sys.platform == "darwin", reason="Avoid error on MacOS arm arch.")
    @pytest.mark.skip("Needs fixing")
    def test_call(self):
        """
        This test simply ensures that the legacy implementation of parameter shift produces
        the same results with the new implementation
        """
        theta_0 = Parameter("theta_0", 0.2)
        theta_1 = Parameter("theta_1", 0.6)
        ham = Hamiltonian.from_pauli_list([("Z0 Z1", 1), ("X1", 1)])

        circ_0 = QuantumCircuit(2)
        circ_0.rx(0, theta_0)
        circ_0.cnot(0, 1)
        circ_0.ry(1, theta_1)

        params = [0.2, 0.6]
        estimator = Estimator(circ_0)
        grad = ParamShift(estimator)

        grads_0 = grad(ham, params)
        print(grads_0)

        circ_1 = QuantumCircuit(2)
        circ_1.rx(0, theta_0)
        circ_1.cnot(0, 1)
        circ_1.ry(1, theta_1)
        circ_1.get_parameter_grads()
        grads_1 = grad_para_shift(circ_1, ham)
        print(grads_1)

        assert np.allclose(grads_0, grads_1, atol=1e-6)
