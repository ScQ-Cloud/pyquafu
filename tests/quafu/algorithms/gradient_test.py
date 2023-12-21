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

import pytest
from quafu.algorithms.estimator import Estimator
from quafu.algorithms.gradients import ParamShift
from quafu.algorithms.hamiltonian import Hamiltonian
from quafu.circuits.quantum_circuit import QuantumCircuit


class TestParamShift:
    @pytest.mark.skipif(
        sys.platform == "darwin", reason="Avoid error on MacOS arm arch."
    )
    def test_call(self):
        ham = Hamiltonian.from_pauli_list([("ZZ", 1), ("XI", 1)])
        circ = QuantumCircuit(2)
        # circ.h(0)
        # circ.h(1)
        circ.rx(0, 0.5)
        circ.cnot(0, 1)
        circ.ry(1, 0.5)

        params = [0.2, 0.6]
        estimator = Estimator(circ)
        grad = ParamShift(estimator)

        grads = grad(ham, params)
        print(grads)
