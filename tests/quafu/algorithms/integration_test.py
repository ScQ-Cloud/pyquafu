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
        eval_answers = heapq.nlargest(2, range(len(probs)), key=probs.__getitem__)

        # Transform to binary string
        num_bit = len(correct_answers[0])
        eval_answers = sorted(
            [bin(eans)[2:].rjust(num_bit, "0") for eans in eval_answers]
        )

        assert eval_answers == sorted(correct_answers)

    def test_ansatz_construction(self):
        num_layers = 2
        print("The test for ansatz.")

        def one_qubits_evolution(pauli):
            h = Hamiltonian.from_pauli_list(
                [
                    (f"{pauli}0", 1),
                    (f"{pauli}1", 1),
                    (f"{pauli}3", 1),
                    (f"{pauli}4", 1),
                ]
            )
            QAOAAnsatz(h, 5, num_layers=num_layers)

        for p in "XYZ":
            one_qubits_evolution(p)

        # test the two qubits evolution
        def two_qubits_evolution(pauli0, pauli1):
            h = Hamiltonian.from_pauli_list(
                [
                    (f"{pauli0}0 {pauli1}1", 1),
                    (f"{pauli0}0 {pauli1}2", 1),
                    (f"{pauli0}0 {pauli1}3", 1),
                    (f"{pauli0}0 {pauli1}4", 1),
                ]
            )
            QAOAAnsatz(h, 5, num_layers=num_layers)

        two_q_paulis = ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"]

        for two_q_pauli in two_q_paulis:
            two_qubits_evolution(two_q_pauli[0], two_q_pauli[1])

        # test the multiple qubits evolution
        hamiltonian_multi = Hamiltonian.from_pauli_list(
            [
                ("X0 Z2 Y3 X4", 1),
                ("X0 Z1 Y3 X4", 1),
                ("X0 Z1 Y2 X4", 1),
                ("X0 Z1 Y2 X3", 1),
            ]
        )
        ansatz_multi = QAOAAnsatz(hamiltonian_multi, 5, num_layers=num_layers)
        ansatz_multi.draw_circuit()

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

        hamiltonian = Hamiltonian.from_pauli_list(
            [("Z0 Z1", 1), ("Z0 Z2", 1), ("Z0 Z3", 1), ("Z0 Z4", 1)]
        )

        ref_mat = np.load("tests/quafu/algorithms/data/qaoa_hamiltonian.npy")
        assert np.array_equal(ref_mat, hamiltonian.get_matrix(5).toarray())
        ansatz = QAOAAnsatz(hamiltonian, 5, num_layers=num_layers)
        ansatz.draw_circuit()

        def cost_func(params, ham, estimator: Estimator):
            return estimator.run(ham, params)

        est = Estimator(ansatz)
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
            [("Z0 Y1", 0.3980), ("Z1", -0.3980), ("Z0 Z1", -0.0113), ("X0 X1", 0.1810)]
        )
        ref_mat = np.load("tests/quafu/algorithms/data/vqe_hamiltonian.npy")
        assert np.array_equal(ref_mat, hamlitonian.get_matrix(2).toarray())
        ansatz = AlterLayeredAnsatz(num_qubits, num_layers)

        def cost_func(params, ham, estimator: Estimator):
            return estimator.run(ham, params)

        est = Estimator(ansatz)
        num_params = ansatz.num_parameters
        assert num_params == 10
        params = 2 * np.pi * np.random.rand(num_params)
        res = minimize(cost_func, params, args=(hamlitonian, est), method="COBYLA")
        print(res)
