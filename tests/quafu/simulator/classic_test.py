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

import unittest

from base import BaseTest

from quafu import QuantumCircuit, simulate


class ClassicalCircuits:
    """Container for reference circuits used by the tests."""

    @staticmethod
    def cif_true():
        qc = QuantumCircuit(2, 2)
        qc.x(0)
        qc.measure([0], [0])
        with qc.cif([0], 1):
            qc.x(1)
        qc.measure([1], [1])
        return qc

    @staticmethod
    def cif_false():
        qc = QuantumCircuit(2)
        qc.measure([0], [0])
        with qc.cif([0], 1):
            qc.x(1)
        qc.measure([1], [1])
        return qc

    @staticmethod
    def cif_list_true():
        qc = QuantumCircuit(3, 3)
        qc.x(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        with qc.cif([0, 1], 3):
            qc.x(2)
        qc.measure([2], [2])
        return qc

    @staticmethod
    def cif_list_false():
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        with qc.cif([0, 1], 0):
            qc.x(2)
        with qc.cif([0, 1], 1):
            qc.x(2)
        with qc.cif([0, 1], 2):
            qc.x(2)
        # skip '11'->3
        with qc.cif([0, 1], 4):
            qc.x(2)
        qc.measure([2], [2])
        return qc

    @staticmethod
    def single_reset():
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.measure([0], [0])
        qc.reset([0])
        qc.measure([0], [1])
        return qc

    @staticmethod
    def multi_reset():
        qc = QuantumCircuit(4)
        qc.x(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        qc.reset([0, 1])
        qc.measure([0, 1], [2, 3])
        return qc


class TestSimulatorClassic(BaseTest):
    """Test C++ simulator"""

    circuit = None
    assertEqual = unittest.TestCase.assertEqual
    assertAlmostEqual = unittest.TestCase.assertAlmostEqual
    assertDictEqual = unittest.TestCase.assertDictEqual
    assertListEqual = unittest.TestCase.assertListEqual
    assertTrue = unittest.TestCase.assertTrue

    def test_cif_true(self):
        self.circuit = ClassicalCircuits.cif_true()
        result = simulate(qc=self.circuit, shots=10)
        probs = result.probabilities
        count = result.count
        self.assertAlmostEqual(probs[0], 0)
        self.assertAlmostEqual(probs[1], 0)
        self.assertAlmostEqual(probs[2], 0)
        self.assertAlmostEqual(probs[3], 1)
        self.assertDictAlmostEqual(count, {"11": 10})

    def test_cif_false(self):
        self.circuit = ClassicalCircuits.cif_false()
        result = simulate(qc=self.circuit, shots=10)
        probs = result.probabilities
        count = result.count
        self.assertAlmostEqual(probs[0], 1)
        self.assertAlmostEqual(probs[1], 0)
        self.assertAlmostEqual(probs[2], 0)
        self.assertAlmostEqual(probs[3], 0)
        self.assertDictAlmostEqual(count, {"00": 10})

    def test_cif_list_true(self):
        self.circuit = ClassicalCircuits.cif_list_true()
        result = simulate(qc=self.circuit, shots=10)
        probs = result.probabilities
        count = result.count
        self.assertAlmostEqual(probs[0], 0)
        self.assertAlmostEqual(probs[1], 0)
        self.assertAlmostEqual(probs[2], 0)
        self.assertAlmostEqual(probs[3], 0)
        self.assertAlmostEqual(probs[4], 0)
        self.assertAlmostEqual(probs[5], 0)
        self.assertAlmostEqual(probs[6], 0)
        self.assertAlmostEqual(probs[7], 1)
        self.assertDictAlmostEqual(count, {"111": 10})

    def test_cif_list_false(self):
        self.circuit = ClassicalCircuits.cif_list_false()
        result = simulate(qc=self.circuit, shots=10)
        probs = result.probabilities
        count = result.count
        self.assertAlmostEqual(probs[0], 0)
        self.assertAlmostEqual(probs[1], 0)
        self.assertAlmostEqual(probs[2], 0)
        self.assertAlmostEqual(probs[3], 0)
        self.assertAlmostEqual(probs[4], 0)
        self.assertAlmostEqual(probs[5], 0)
        self.assertAlmostEqual(probs[6], 1)
        self.assertAlmostEqual(probs[7], 0)
        self.assertDictAlmostEqual(count, {"110": 10})

    def test_single_reset(self):
        self.circuit = ClassicalCircuits.single_reset()
        result = simulate(qc=self.circuit, shots=10)
        probs = result.probabilities
        count = result.count
        self.assertAlmostEqual(probs[0], 1)
        self.assertAlmostEqual(probs[1], 0)
        self.assertDictAlmostEqual(count, {"10": 10})

    def test_multi_reset(self):
        self.circuit = ClassicalCircuits.multi_reset()
        result = simulate(qc=self.circuit, shots=10)
        probs = result.probabilities
        count = result.count
        self.assertAlmostEqual(probs[0], 1)
        self.assertAlmostEqual(probs[1], 0)
        self.assertAlmostEqual(probs[2], 0)
        self.assertAlmostEqual(probs[3], 0)
        self.assertDictAlmostEqual(count, {"1100": 10})
