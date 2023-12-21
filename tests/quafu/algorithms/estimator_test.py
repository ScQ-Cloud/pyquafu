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

import math
import sys
from unittest.mock import patch

import numpy as np
import pytest
from quafu.algorithms.estimator import Estimator
from quafu.algorithms.hamiltonian import Hamiltonian
from quafu.circuits.quantum_circuit import QuantumCircuit
from quafu.tasks.tasks import Task

from quafu import ExecResult

MOCK_RES_DICT = {
    "measure": "None",
    "openqasm": 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[136];\ncreg c[5];\nry(-1.5707963267948966) q[32];\nrz(-3.141592653589793) q[33];\nrz(-3.141592653589793) q[44];\nry(-1.5707963267948966) q[45];\nrz(-3.141592653589793) q[54];\nbarrier q[54],q[32],q[44],q[45],q[33];\nmeasure q[54] -> c[0];\nmeasure q[32] -> c[1];\nmeasure q[44] -> c[2];\nmeasure q[45] -> c[3];\nmeasure q[33] -> c[4];\n',
    "raw": '{"00000": 192, "01000": 180, "00010": 186, "01010": 224, "01110": 20, "11010": 19, "00011": 14, "10010": 19, "00111": 1, "01011": 15, "11000": 17, "00001": 9, "10000": 28, "10100": 5, "01101": 1, "00100": 18, "00110": 15, "01100": 18, "11011": 2, "01001": 9, "11111": 1, "11001": 2, "10001": 2, "10110": 2, "11110": 1}',
    "res": '{"00000": 192, "01000": 180, "00010": 186, "01010": 224, "01110": 20, "11010": 19, "00011": 14, "10010": 19, "00111": 1, "01011": 15, "11000": 17, "00001": 9, "10000": 28, "10100": 5, "01101": 1, "00100": 18, "00110": 15, "01100": 18, "11011": 2, "01001": 9, "11111": 1, "11001": 2, "10001": 2, "10110": 2, "11110": 1}',
    "status": 2,
    "task_id": "30EE7D5035E7CE02",
    "task_name": "",
}
TEST_EXE_RES = ExecResult(MOCK_RES_DICT)


class TestEstimator:
    """Test class of Estimator"""

    def build_circuit(self):
        """Build a random circuit for testing purpose"""
        circ = QuantumCircuit(5)

        for i in range(5):
            if i % 2 == 0:
                circ.h(i)

        circ.cnot(0, 1)
        circ.draw_circuit()
        measures = list(range(5))
        circ.measure(measures)
        test_ising = Hamiltonian(
            ["IIIZZ", "ZZIII", "IZZII", "ZIIIZ"], np.array([1, 1, 1, 1])
        )
        return circ, test_ising

    @patch("quafu.users.userapi.User._load_account_token", autospec=True)
    @patch("quafu.users.userapi.User.get_available_backends", autospec=True)
    @patch("quafu.tasks.tasks.Task.send", autospec=True)
    def test_run(self, mock_send, mock_backends, mock_load_account):
        """Test Estimator.run"""
        mock_send.return_value = TEST_EXE_RES
        mock_backends.return_value = {"ScQ-P10": None}
        circ, test_ising = self.build_circuit()
        estimator = Estimator(circ, backend="ScQ-P10")
        expectation = estimator.run(test_ising, None)
        task = Task()
        res_org, obsexp_org = task.submit(circ, test_ising.to_legacy_quafu_pauli_list())
        assert expectation == sum(obsexp_org)

    @pytest.mark.skipif(
        sys.platform == "darwin", reason="Avoid error on MacOS arm arch."
    )
    def test_run_sim(self):
        circ, test_ising = self.build_circuit()
        estimator = Estimator(circ)
        expectation = estimator.run(test_ising, None)
        assert math.isclose(expectation, 1.0)
