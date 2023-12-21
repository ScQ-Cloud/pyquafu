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
