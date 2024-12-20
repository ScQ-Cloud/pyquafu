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
"""TODO: test of ansatz needs improvement once ansatz has more featuers"""

from quafu.algorithms.ansatz import AlterLayeredAnsatz, QAOAAnsatz
from quafu.algorithms.hamiltonian import Hamiltonian, PauliOp


class TestQAOACircuit:
    TEST_HAM = Hamiltonian([PauliOp("Z0 Z1"), PauliOp("Z2 Z3"), PauliOp("Z1 Z2"), PauliOp("Z0 Z3")])

    def test_build(self):
        qaoa = QAOAAnsatz(self.TEST_HAM, 4)
        print("\n ::: testing ::: \n")
        qaoa.draw_circuit()

    def test_update_params(self):
        pass


class TestAlterLayeredAnsatz:
    def test_build(self):
        AlterLayeredAnsatz(4, 4)
