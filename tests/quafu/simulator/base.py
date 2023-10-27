class BaseTest:
    def assertDictAlmostEqual(self, dict1, dict2, delta=None, places=None, default_value=-1):
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
