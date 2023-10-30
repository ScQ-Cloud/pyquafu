import sys
import pytest
from quafu import QuantumCircuit
from quafu import simulate
from base import BaseTest
import unittest
import numpy as np

class BellCircuits:
    """Container for reference circuits used by the tests."""

    @staticmethod
    def bell_measure_atlast():
        """Return a Bell circuit."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0,1])
        return qc
    
    @staticmethod
    def bell_measure_normal():
        """Return a Bell circuit."""
        qc = QuantumCircuit(3, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0,1])
        qc.h(2)
        return qc

    @staticmethod
    def bell_no_measure():
        """Return a Bell circuit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

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
        print(probs)
        self.assertAlmostEqual(probs[0], 1/2)
        self.assertAlmostEqual(probs[1], 0)
        self.assertAlmostEqual(probs[2], 0)
        self.assertAlmostEqual(probs[3], 1/2)
        self.assertDictAlmostEqual(count, {})
    
    @pytest.mark.skipif(
        sys.platform == "darwin", reason="Avoid error on MacOS arm arch."
    )
    def test_measure_atlast_collapse(self):
        """Test final measurement statement"""
        self.circuit = BellCircuits.bell_measure_atlast()
        result = simulate(qc=self.circuit)
        probs = result.probabilities
        self.assertAlmostEqual(probs[0], 1/2)
        self.assertAlmostEqual(probs[1], 0)
        self.assertAlmostEqual(probs[2], 0)
        self.assertAlmostEqual(probs[3], 1/2)
    
    @pytest.mark.skipif(
        sys.platform == "darwin", reason="Avoid error on MacOS arm arch."
    )
    def test_measure_normal_collapse(self):
        """Test final measurement statement"""
        self.circuit = BellCircuits.bell_measure_normal()
        result = simulate(qc=self.circuit, shots=1)
        probs = result.probabilities
        print(probs)
        diff_00 = np.linalg.norm(np.array([1, 0, 0, 0]) - probs) ** 2
        diff_11 = np.linalg.norm(np.array([0, 0, 0, 1]) - probs) ** 2
        print(diff_00)
        print(diff_11)
        success = np.allclose([diff_00, diff_11], [0, 2]) or np.allclose([diff_00, diff_11], [2, 0])
        # state is 1/sqrt(2)|00> + 1/sqrt(2)|11>, up to a global phase
        self.assertTrue(success)
    
    def test_x(self):
        pass