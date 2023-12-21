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

from quafu.qfasm.qfasm_convertor import qasm_to_quafu

from quafu import simulate


class BaseTest:
    def assertDictAlmostEqual(
        self, dict1, dict2, delta=None, places=None, default_value=-1
    ):
        """
        Assert two dictionaries with numeric values are almost equal.

        Args:
            dict1 (dict): a dictionary.
            dict2 (dict): a dictionary.
            delta (number): threshold for comparison (defaults to 1e-8).
            msg (str): return a custom message on failure.
            places (int): number of decimal places for comparison.
            default_value (number): default value for missing keys.
        """

        def valid_comparison(value):
            """compare value to delta, within places accuracy"""
            if places is not None:
                return round(value, places) == 0
            else:
                return value < delta

        # Check arguments.
        if dict1 == dict2:
            return
        if places is None and delta is None:
            delta = delta or 1e-8

        # Compare all keys in both dicts, populating error_msg.
        for key in set(dict1.keys()) | set(dict2.keys()):
            val1 = dict1.get(key, default_value)
            val2 = dict2.get(key, default_value)
            if not valid_comparison(abs(val1 - val2)):
                raise Exception("Dict not equal")


class TestClassicOp(BaseTest):
    assertEqual = unittest.TestCase.assertEqual
    assertAlmostEqual = unittest.TestCase.assertAlmostEqual
    assertDictEqual = unittest.TestCase.assertDictEqual
    assertListEqual = unittest.TestCase.assertListEqual
    assertTrue = unittest.TestCase.assertTrue

    def test_single_reset(self):
        qasm = "qreg q[2];creg c[2];x q; reset q[0]; measure q[0]->c[0];"
        circ = qasm_to_quafu(openqasm=qasm)
        assert circ.instructions[-2].name == "reset"
        assert circ.instructions[-2].pos == [0]

    def test_multi_reset(self):
        qasm = "qreg q[2];creg c[2];x q; reset q; measure q[0]->c[0];"
        circ = qasm_to_quafu(openqasm=qasm)
        assert circ.instructions[-2].name == "reset"
        assert circ.instructions[-2].pos == [0, 1]

    def test_cif_single(self):
        qasm = """
        qreg q[2];
        creg c[2];
        x q[0];
        measure q[0]->c[0];
        if(c[0] == 1)
            x q[1];
        measure q[1]->c[1];
        """
        circ = qasm_to_quafu(openqasm=qasm)
        assert circ.instructions[-2].name == "cif"
        assert circ.instructions[-2].cbits == [0]
        assert circ.instructions[-2].condition == 1
        assert len(circ.instructions[-2].instructions) == 1
        result = simulate(qc=circ, shots=10)
        probs = result.probabilities
        count = result.count
        self.assertAlmostEqual(probs[0], 0)
        self.assertAlmostEqual(probs[1], 0)
        self.assertAlmostEqual(probs[2], 0)
        self.assertAlmostEqual(probs[3], 1)
        self.assertDictAlmostEqual(count, {"11": 10})

    def test_cif_multi(self):
        qasm = """
        qreg q[2];
        qreg m[2];
        creg c[2];
        creg n[2];
        x q;
        measure q->c;
        if(c == 3)
            x m;
        measure m->n;
        """
        circ = qasm_to_quafu(openqasm=qasm)
        assert circ.instructions[-2].name == "cif"
        assert circ.instructions[-2].cbits == [0, 1]
        assert circ.instructions[-2].condition == 3
        assert len(circ.instructions[-2].instructions) == 2  # x m[0]; x m[1];
        result = simulate(qc=circ, shots=10)
        probs = result.probabilities
        count = result.count
        self.assertAlmostEqual(probs[0], 0)
        self.assertAlmostEqual(probs[1], 0)
        self.assertAlmostEqual(probs[2], 0)
        self.assertAlmostEqual(probs[15], 1)
        self.assertDictAlmostEqual(count, {"1111": 10})
