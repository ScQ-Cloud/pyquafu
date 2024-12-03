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
"""Quafu parameter shift."""

from typing import List, Optional

import numpy as np

from ..estimator import Estimator
from ..hamiltonian import Hamiltonian
from .gradiant import grad_para_shift


class ParamShift:
    """Parameter shift rule to calculate gradients"""

    def __init__(self, estimator: Estimator) -> None:
        self._est = estimator

    def __call__(self, obs: Hamiltonian, params: List[float], cache_key: Optional[str] = None):
        """Calculate gradients using paramshift.

        Args:
            estimator (Estimator): estimator to calculate expectation values
            params (List[float]): params to optimize
        """
        if self._est._backend != "sim":
            return self.grad(obs, params, cache_key=cache_key)
        return self.new_grad(obs, params)

    def _gen_param_shift_vals(self, params):
        """Given a param list with n values, replicate to 2*n param list"""
        num_vals = len(params)
        params = np.array(params)
        offsets = np.identity(num_vals)
        plus_params = params + offsets * np.pi / 2
        minus_params = params - offsets * np.pi / 2
        return plus_params.tolist() + minus_params.tolist()

    def grad(self, obs: Hamiltonian, params: List[float], cache_key: Optional[str] = None):
        """grad.

        Args:
            obs (Hamiltonian): obs
            params (List[float]): params
            cache_key: cache prefix, currently the sample id in a batch
        """
        shifted_params_lists = self._gen_param_shift_vals(params)

        res = np.zeros(len(shifted_params_lists))
        for i, shifted_params in enumerate(shifted_params_lists):
            final_cache_key = None
            if cache_key is not None:
                # parameters is uniquely determined by
                # <sample-id-in-the-batch><order-in-shifted-parameters>
                final_cache_key = cache_key + str(i)
            res[i] = self._est.run(obs, shifted_params, cache_key=final_cache_key)

        num_shift_params = len(res)
        return (res[: num_shift_params // 2] - res[num_shift_params // 2 :]) / 2

    def new_grad(self, obs: Hamiltonian, params: List[float]):
        """Calculate the gradients of given the circuit based on the parameter shift rule
        Args:
            obs (Hamiltonian): observables for measurement.
            params (List[float]): parameters to apply to the circuit.
        """
        self._est._circ._update_params(params)
        return grad_para_shift(self._est._circ, obs)
