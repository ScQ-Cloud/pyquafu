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
"""Quafu parameter shift"""

from typing import List

import numpy as np

from ..estimator import Estimator
from ..hamiltonian import Hamiltonian


class ParamShift:
    """Parameter shift rule to calculate gradients"""

    def __init__(self, estimator: Estimator) -> None:
        self._est = estimator

    def __call__(self, obs: Hamiltonian, params: List[float]):
        """Calculate gradients using paramshift.

        Args:
            estimator (Estimator): estimator to calculate expectation values
            params (List[float]): params to optimize
        """
        return self.grad(obs, params)

    def _gen_param_shift_vals(self, params):
        """Given a param list with n values, replicate to 2*n param list"""
        num_vals = len(params)
        params = np.array(params)
        offsets = np.identity(num_vals)
        plus_params = params + offsets * np.pi / 2
        minus_params = params - offsets * np.pi / 2
        return plus_params.tolist() + minus_params.tolist()

    def grad(self, obs: Hamiltonian, params: List[float]):
        """grad.

        Args:
            obs (Hamiltonian): obs
            params (List[float]): params
        """
        shifted_params_lists = self._gen_param_shift_vals(params)

        res = np.zeros(len(shifted_params_lists))
        for i, shifted_params in enumerate(shifted_params_lists):
            res[i] = self._est.run(obs, shifted_params)

        num_shift_params = len(res)
        grads = (res[: num_shift_params // 2] - res[num_shift_params // 2 :]) / 2
        return grads
