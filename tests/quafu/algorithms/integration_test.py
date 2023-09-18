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
from typing import List
import numpy as np
import pytest
from quafu.algorithms import Hamiltonian, QAOAAnsatz, Estimator
from quafu import simulate
from scipy.optimize import minimize
import heapq

from quafu.algorithms.ansatz import AlterLayeredAnsatz


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
        hamlitonian = Hamiltonian.from_pauli_list(
            [("IIIZZ", 1), ("IIZIZ", 1), ("IZIIZ", 1), ("ZIIIZ", 1)]
        )
        ref_mat = np.load("tests/quafu/algorithms/test_hamiltonian.npy")
        assert np.array_equal(ref_mat, hamlitonian.get_matrix())
        ansatz = QAOAAnsatz(hamlitonian, num_layers=num_layers)
        ansatz.draw_circuit()

        def cost_func(params, ham, estimator: Estimator):
            cost = estimator.run(ham, params)
            return cost

        est = Estimator(ansatz)
        # params = 2 * np.pi * np.random.rand(num_layers * 2)
        params = 2 * np.pi * np.random.rand(ansatz.num_parameters)
        res = minimize(cost_func, params, args=(hamlitonian, est), method="COBYLA")
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
        # TODO: ref mat
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
