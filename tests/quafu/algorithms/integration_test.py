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

import heapq
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pytest
from quafu.algorithms import Estimator, Hamiltonian, QAOAAnsatz
from quafu.algorithms.ansatz import AlterLayeredAnsatz
from scipy.optimize import minimize

from quafu import simulate


class TestQAOA:
    """Example of a QAOA algorithm to solve maxcut problem"""

    def check_solution(self, probs: List[float], correct_answers: List[str]):
        """Check the MAX-CUT solution
        Args:
            probs: probabilities of simulation result.
            correct_answers: List of optimal solutions in str format.
        """

        # Get top # correct_answers probabilities and find indexes
        num_answers = len(correct_answers)
        eval_answers = heapq.nlargest(2, range(len(probs)), key=probs.__getitem__)

        # Transform to binary string
        num_bit = len(correct_answers[0])
        eval_answers = sorted(
            [bin(eans)[2:].rjust(num_bit, "0") for eans in eval_answers]
        )

        assert eval_answers == sorted(correct_answers)

    @pytest.mark.skipif(
        sys.platform == "darwin", reason="Avoid error on MacOS arm arch."
    )
    def test_run(self):
        """
        A simple graph with 5 nodes and connected as below
            a -> b -> c
            d ---^
            e ---^
        We use 5 qubits to solve this problem and
        the measurement result "10000" and "01111"
        should have the highest probability
        """
        num_layers = 2
        print("The test for ansatz.")

        # test the zero qubit evolution
        hamiltonian__ = Hamiltonian.from_pauli_list(
            [("IIIII", 1), ("IIIII", 1), ("IIIII", 1), ("IIIII", 1)]
        )
        ansatz__ = QAOAAnsatz(hamiltonian__, num_layers=num_layers)
        ansatz__.draw_circuit()

        # test the single qubit evolution
        hamiltonian_x = Hamiltonian.from_pauli_list(
            [("IIIIX", 1), ("IIIXI", 1), ("IXIII", 1), ("XIIII", 1)]
        )
        ansatz_x = QAOAAnsatz(hamiltonian_x, num_layers=num_layers)
        ansatz_x.draw_circuit()
        hamiltonian_y = Hamiltonian.from_pauli_list(
            [("IIIIY", 1), ("IIIYI", 1), ("IYIII", 1), ("YIIII", 1)]
        )
        ansatz_y = QAOAAnsatz(hamiltonian_y, num_layers=num_layers)
        ansatz_y.draw_circuit()
        hamiltonian_z = Hamiltonian.from_pauli_list(
            [("IIIIZ", 1), ("IIIZI", 1), ("IZIII", 1), ("ZIIII", 1)]
        )
        ansatz_z = QAOAAnsatz(hamiltonian_z, num_layers=num_layers)
        ansatz_z.draw_circuit()

        # test the two qubits evolution
        hamiltonian_xx = Hamiltonian.from_pauli_list(
            [("IIIXX", 1), ("IIXIX", 1), ("IXIIX", 1), ("XIIIX", 1)]
        )
        ansatz_xx = QAOAAnsatz(hamiltonian_xx, num_layers=num_layers)
        ansatz_xx.draw_circuit()
        hamiltonian_xy = Hamiltonian.from_pauli_list(
            [("IIIYX", 1), ("IIYIX", 1), ("IYIIX", 1), ("YIIIX", 1)]
        )
        ansatz_xy = QAOAAnsatz(hamiltonian_xy, num_layers=num_layers)
        ansatz_xy.draw_circuit()
        hamiltonian_xz = Hamiltonian.from_pauli_list(
            [("IIIXZ", 1), ("IIZIX", 1), ("IZIIX", 1), ("ZIIIX", 1)]
        )
        ansatz_xz = QAOAAnsatz(hamiltonian_xz, num_layers=num_layers)
        ansatz_xz.draw_circuit()
        hamiltonian_yx = Hamiltonian.from_pauli_list(
            [("IIIXY", 1), ("IIXIY", 1), ("IXIIY", 1), ("XIIIY", 1)]
        )
        ansatz_yx = QAOAAnsatz(hamiltonian_yx, num_layers=num_layers)
        ansatz_yx.draw_circuit()
        hamiltonian_yy = Hamiltonian.from_pauli_list(
            [("IIIYY", 1), ("IIYIY", 1), ("IYIIY", 1), ("YIIIY", 1)]
        )
        ansatz_yy = QAOAAnsatz(hamiltonian_yy, num_layers=num_layers)
        ansatz_yy.draw_circuit()
        hamiltonian_yz = Hamiltonian.from_pauli_list(
            [("IIIZY", 1), ("IIZIY", 1), ("IZIIY", 1), ("ZIIIY", 1)]
        )
        ansatz_yz = QAOAAnsatz(hamiltonian_yz, num_layers=num_layers)
        ansatz_yz.draw_circuit()
        hamiltonian_zx = Hamiltonian.from_pauli_list(
            [("IIIXZ", 1), ("IIXIZ", 1), ("IXIIZ", 1), ("XIIIZ", 1)]
        )
        ansatz_zx = QAOAAnsatz(hamiltonian_zx, num_layers=num_layers)
        ansatz_zx.draw_circuit()
        hamiltonian_zy = Hamiltonian.from_pauli_list(
            [("IIIYZ", 1), ("IIYIZ", 1), ("IYIIZ", 1), ("YIIIZ", 1)]
        )
        ansatz_zy = QAOAAnsatz(hamiltonian_zy, num_layers=num_layers)
        ansatz_zy.draw_circuit()
        hamiltonian_zz = Hamiltonian.from_pauli_list(
            [("IIIZZ", 1), ("IIZIZ", 1), ("IZIIZ", 1), ("ZIIIZ", 1)]
        )
        ansatz_zz = QAOAAnsatz(hamiltonian_zz, num_layers=num_layers)
        ansatz_zz.draw_circuit()

        # test the multiple qubits evolution
        hamiltonian_multi = Hamiltonian.from_pauli_list(
            [("XYZIX", 1), ("XYIZX", 1), ("XIYZX", 1), ("IXYZX", 1)]
        )
        ansatz_multi = QAOAAnsatz(hamiltonian_multi, num_layers=num_layers)
        ansatz_multi.draw_circuit()
        # ansatz_multi.plot_circuit(title='MULTI QUBITS')
        # plt.show()

        hamiltonian = Hamiltonian.from_pauli_list(
            [("IIIZZ", 1), ("IIZIZ", 1), ("IZIIZ", 1), ("ZIIIZ", 1)]
        )
        ref_mat = np.load("tests/quafu/algorithms/data/qaoa_hamiltonian.npy")
        # ref_mat = np.load("data/qaoa_hamiltonian.npy")
        assert np.array_equal(ref_mat, hamiltonian.get_matrix())
        ansatz = QAOAAnsatz(hamiltonian, num_layers=num_layers)
        ansatz.draw_circuit()

        def cost_func(params, ham, estimator: Estimator):
            cost = estimator.run(ham, params)
            return cost

        est = Estimator(ansatz)
        # params = 2 * np.pi * np.random.rand(num_layers * 2)
        params = 2 * np.pi * np.random.rand(ansatz.num_parameters)
        res = minimize(cost_func, params, args=(hamiltonian, est), method="COBYLA")
        print(res)
        ansatz.measure(list(range(5)))
        ansatz.draw_circuit()
        probs = simulate(ansatz).probabilities
        print(probs)
        self.check_solution(list(probs), ["10000", "01111"])
        print("::: PASSED ::: Find Correct Answers:: 01111 and 10000")


class TestVQE:
    @pytest.mark.skipif(
        sys.platform == "darwin", reason="Avoid error on MacOS arm arch."
    )
    def test_run(self):
        """A sample VQE algorithm"""

        num_layers = 4
        num_qubits = 2
        hamlitonian = Hamiltonian.from_pauli_list(
            [("YZ", 0.3980), ("ZI", -0.3980), ("ZZ", -0.0113), ("XX", 0.1810)]
        )
        ref_mat = np.load("tests/quafu/algorithms/data/vqe_hamiltonian.npy")
        assert np.array_equal(ref_mat, hamlitonian.get_matrix())
        ansatz = AlterLayeredAnsatz(num_qubits, num_layers)

        def cost_func(params, ham, estimator: Estimator):
            cost = estimator.run(ham, params)
            return cost

        est = Estimator(ansatz)
        num_params = ansatz.num_parameters
        assert num_params == 10
        params = 2 * np.pi * np.random.rand(num_params)
        res = minimize(cost_func, params, args=(hamlitonian, est), method="COBYLA")
        print(res)
