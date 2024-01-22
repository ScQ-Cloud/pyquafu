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
import unittest

import numpy as np
import pytest
from base import BaseTest

from quafu import QuantumCircuit, simulate


class BellCircuits:
    """Container for reference circuits used by the tests."""

    @staticmethod
    def bell_measure_atlast():
        """Return a Bell circuit."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1])
        return qc

    @staticmethod
    def bell_measure_normal():
        """Return a Bell circuit."""
        qc = QuantumCircuit(3, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1])
        qc.h(2)
        return qc

    @staticmethod
    def bell_no_measure():
        """Return a Bell circuit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        return qc


class BasicCircuits:
    """Container for reference circuits used by the tests."""

    @staticmethod
    def singleQgate_measure_atlast():
        qc = QuantumCircuit(2, 2)
        qc.x(0)
        qc.x(1)
        qc.measure([0, 1])
        return qc

    @staticmethod
    def singleQgate_no_measure():
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.x(1)
        return qc

    @staticmethod
    def singleQgate_measure_normal():
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.measure([0], [0])
        qc.x(1)
        qc.measure([1], [1])
        return qc

    @staticmethod
    def multiQgate_measure_atlast():
        qc = QuantumCircuit(2, 2)
        qc.x(0)
        qc.cx(0, 1)
        qc.measure([0, 1])
        return qc

    @staticmethod
    def multiQgate_no_measure():
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.cx(0, 1)
        return qc

    @staticmethod
    def multiQgate_measure_normal():
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.measure([0], [0])
        qc.cx(0, 1)
        qc.measure([1], [1])
        return qc

    @staticmethod
    def any_cbit_measure():
        qc = QuantumCircuit(4, 4)
        qc.x(0)
        qc.x(1)
        qc.measure([1, 2], [1, 0])
        qc.measure([3, 0], [2, 3])
        return qc

    @staticmethod
    def after_measure():
        qc = QuantumCircuit(2, 22)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0], [0])
        qc.measure([1], [1])
        qc.reset([0, 1])
        return qc


class TestSimulatorBasis(BaseTest):
    """Test C++ simulator"""

    circuit = None
    assertEqual = unittest.TestCase.assertEqual
    assertAlmostEqual = unittest.TestCase.assertAlmostEqual
    assertDictEqual = unittest.TestCase.assertDictEqual
    assertListEqual = unittest.TestCase.assertListEqual
    assertTrue = unittest.TestCase.assertTrue

    @pytest.mark.skipif(
        sys.platform == "darwin", reason="Avoid error on MacOS arm arch."
    )
    def test_simulate(self):
        self.circuit = BellCircuits.bell_no_measure()
        result = simulate(qc=self.circuit)
        probs = result.probabilities
        count = result.count
        self.assertAlmostEqual(probs[0], 1 / 2)
        self.assertAlmostEqual(probs[1], 0)
        self.assertAlmostEqual(probs[2], 0)
        self.assertAlmostEqual(probs[3], 1 / 2)
        self.assertDictAlmostEqual(count, {})

    @pytest.mark.skipif(
        sys.platform == "darwin", reason="Avoid error on MacOS arm arch."
    )
    def test_measure_atlast_collapse(self):
        """Test final measurement statement"""
        self.circuit = BellCircuits.bell_measure_atlast()
        result = simulate(qc=self.circuit)
        probs = result.probabilities
        self.assertAlmostEqual(probs[0], 1 / 2)
        self.assertAlmostEqual(probs[1], 0)
        self.assertAlmostEqual(probs[2], 0)
        self.assertAlmostEqual(probs[3], 1 / 2)

    @pytest.mark.skipif(
        sys.platform == "darwin", reason="Avoid error on MacOS arm arch."
    )
    def test_measure_normal_collapse(self):
        """Test normal measurement statement"""
        self.circuit = BellCircuits.bell_measure_normal()
        result = simulate(qc=self.circuit, shots=1)
        probs = result.probabilities
        diff_00 = np.linalg.norm(np.array([1, 0, 0, 0]) - probs) ** 2
        diff_11 = np.linalg.norm(np.array([0, 0, 0, 1]) - probs) ** 2
        success = np.allclose([diff_00, diff_11], [0, 2]) or np.allclose(
            [diff_00, diff_11], [2, 0]
        )
        # state is 1/sqrt(2)|00> + 1/sqrt(2)|11>, up to a global phase
        self.assertTrue(success)

    def test_singleQgate_measure_atlast(self):
        self.circuit = BasicCircuits.singleQgate_measure_atlast()
        result = simulate(qc=self.circuit, shots=1)
        probs = result.probabilities
        counts = result.count
        self.assertAlmostEqual(probs[0], 0)
        self.assertAlmostEqual(probs[1], 0)
        self.assertAlmostEqual(probs[2], 0)
        self.assertAlmostEqual(probs[3], 1)
        self.assertDictAlmostEqual(counts, {"11": 1})

    def test_singleQgate_no_measure(self):
        self.circuit = BasicCircuits.singleQgate_no_measure()
        result = simulate(qc=self.circuit, shots=1)
        probs = result.probabilities
        counts = result.count
        self.assertAlmostEqual(probs[0], 0)
        self.assertAlmostEqual(probs[1], 0)
        self.assertAlmostEqual(probs[2], 0)
        self.assertAlmostEqual(probs[3], 1)
        self.assertDictAlmostEqual(counts, {})

    def test_singleQgate_measure_normal(self):
        self.circuit = BasicCircuits.singleQgate_measure_normal()
        result = simulate(qc=self.circuit, shots=10)
        probs = result.probabilities
        counts = result.count
        self.assertAlmostEqual(probs[0], 0)
        self.assertAlmostEqual(probs[1], 0)
        self.assertAlmostEqual(probs[2], 0)
        self.assertAlmostEqual(probs[3], 1)
        self.assertDictAlmostEqual(counts, {"11": 10})

    def test_multiQgate_measure_atlast(self):
        self.circuit = BasicCircuits.multiQgate_measure_atlast()
        result = simulate(qc=self.circuit, shots=10)
        probs = result.probabilities
        counts = result.count
        self.assertAlmostEqual(probs[0], 0)
        self.assertAlmostEqual(probs[1], 0)
        self.assertAlmostEqual(probs[2], 0)
        self.assertAlmostEqual(probs[3], 1)
        self.assertDictAlmostEqual(counts, {"11": 10})

    def test_multiQgate_no_measure(self):
        self.circuit = BasicCircuits.multiQgate_no_measure()
        result = simulate(qc=self.circuit, shots=1)
        probs = result.probabilities
        counts = result.count
        self.assertAlmostEqual(probs[0], 0)
        self.assertAlmostEqual(probs[1], 0)
        self.assertAlmostEqual(probs[2], 0)
        self.assertAlmostEqual(probs[3], 1)
        self.assertDictAlmostEqual(counts, {})

    def test_multiQgate_measure_normal(self):
        self.circuit = BasicCircuits.multiQgate_measure_normal()
        result = simulate(qc=self.circuit, shots=10)
        probs = result.probabilities
        counts = result.count
        self.assertAlmostEqual(probs[0], 0)
        self.assertAlmostEqual(probs[1], 0)
        self.assertAlmostEqual(probs[2], 0)
        self.assertAlmostEqual(probs[3], 1)
        self.assertDictAlmostEqual(counts, {"11": 10})

    def test_anycbit_measure(self):
        self.circuit = BasicCircuits.any_cbit_measure()
        result = simulate(qc=self.circuit, shots=10)
        probs = result.probabilities
        counts = result.count
        print(probs)
        self.assertAlmostEqual(probs[5], 1)  # 0101
        self.assertDictAlmostEqual(counts, {"0101": 10})

    def test_after_measure(self):
        self.circuit = BasicCircuits.after_measure()
        result = simulate(qc=self.circuit, shots=10)
        probs = result.probabilities
        diff_00 = np.linalg.norm(np.array([1, 0, 0, 0]) - probs) ** 2
        diff_11 = np.linalg.norm(np.array([0, 0, 0, 1]) - probs) ** 2
        success = np.allclose([diff_00, diff_11], [0, 2]) or np.allclose(
            [diff_00, diff_11], [2, 0]
        )
        self.assertTrue(success)


class TestCliffordSimulatorBasis(BaseTest):
    """Test C++ Clifford simulator"""

    circuit = None
    assertEqual = unittest.TestCase.assertEqual
    assertAlmostEqual = unittest.TestCase.assertAlmostEqual
    assertDictEqual = unittest.TestCase.assertDictEqual
    assertListEqual = unittest.TestCase.assertListEqual
    assertTrue = unittest.TestCase.assertTrue

    @pytest.mark.skipif(
        sys.platform == "darwin", reason="Avoid error on MacOS arm arch."
    )
    def test_simulate(self):
        print("test_simulate")
        self.circuit = BellCircuits.bell_no_measure()
        result = simulate(
            qc=self.circuit, simulator="qfvm_clifford", output="count_dict"
        )
        count = result.count
        self.assertDictAlmostEqual(count, {})

    def test_singleQgate_measure_atlast(self):
        self.circuit = BasicCircuits.singleQgate_measure_atlast()
        result = simulate(
            qc=self.circuit, shots=1, simulator="qfvm_clifford", output="count_dict"
        )
        counts = result.count
        self.assertDictAlmostEqual(counts, {"11": 1})

    def test_singleQgate_no_measure(self):
        self.circuit = BasicCircuits.singleQgate_no_measure()
        result = simulate(
            qc=self.circuit, shots=1, simulator="qfvm_clifford", output="count_dict"
        )
        counts = result.count
        self.assertDictAlmostEqual(counts, {})

    def test_singleQgate_measure_normal(self):
        self.circuit = BasicCircuits.singleQgate_measure_normal()
        result = simulate(
            qc=self.circuit, shots=10, simulator="qfvm_clifford", output="count_dict"
        )
        counts = result.count
        self.assertDictAlmostEqual(counts, {"11": 10})

    def test_multiQgate_measure_atlast(self):
        self.circuit = BasicCircuits.multiQgate_measure_atlast()
        result = simulate(
            qc=self.circuit, shots=10, simulator="qfvm_clifford", output="count_dict"
        )
        counts = result.count
        self.assertDictAlmostEqual(counts, {"11": 10})

    def test_multiQgate_no_measure(self):
        self.circuit = BasicCircuits.multiQgate_no_measure()
        result = simulate(
            qc=self.circuit, shots=1, simulator="qfvm_clifford", output="count_dict"
        )
        counts = result.count
        self.assertDictAlmostEqual(counts, {})

    def test_multiQgate_measure_normal(self):
        self.circuit = BasicCircuits.multiQgate_measure_normal()
        result = simulate(
            qc=self.circuit, shots=10, simulator="qfvm_clifford", output="count_dict"
        )
        counts = result.count
        self.assertDictAlmostEqual(counts, {"11": 10})

    def test_anycbit_measure(self):
        self.circuit = BasicCircuits.any_cbit_measure()
        result = simulate(
            qc=self.circuit, shots=10, simulator="qfvm_clifford", output="count_dict"
        )
        counts = result.count
        self.assertDictAlmostEqual(counts, {"0101": 10})
