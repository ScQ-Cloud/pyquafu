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

from unittest.mock import patch
from quafu import ExecResult
from quafu.algorithms.estimator import Estimator

from quafu.circuits.quantum_circuit import QuantumCircuit
from quafu.tasks.tasks import Task

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

    @patch("quafu.tasks.tasks.Task.send")
    def test_run(self, mock_send):
        """Test Estimator.run"""
        mock_send.return_value = TEST_EXE_RES
        circ = QuantumCircuit(5)

        for i in range(5):
            if i % 2 == 0:
                circ.h(i)

        circ.draw_circuit()
        measures = list(range(5))
        circ.measure(measures)
        test_ising = [["X", [i]] for i in range(5)]
        test_ising.extend([["ZZ", [i, i + 1]] for i in range(4)])

        estimator = Estimator(circ, backend="ScQ-P10")
        res, obsexp = estimator.run(test_ising)
        task = Task()
        res_org, obsexp_org = task.submit(circ, test_ising)
        assert res == res_org and obsexp == obsexp_org