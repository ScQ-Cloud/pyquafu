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
"""quafu PyTorch quantum layer"""

from typing import Optional

import numpy as np
import torch
from quafu.algorithms.estimator import Estimator

from quafu import QuantumCircuit

from ..gradients import compute_vjp, jacobian, run_circ


# TODO(zhaoyilun): impl a ABC for transformers
class TorchTransformer:
    @staticmethod
    def init_weights(shape):
        """Return torch gradient tensor with specified shape"""
        return torch.randn(*shape, requires_grad=True, dtype=torch.double)

    # TODO(zhaoyilun): docstrings
    @staticmethod
    def execute(
        circ: QuantumCircuit,
        parameters: torch.Tensor,
        run_fn=run_circ,
        grad_fn=None,
        method="internal",
        estimator: Optional[Estimator] = None,
    ):
        """execute.

        Args:
            circ:
            run_fn:
            grad_fn:
        """

        kwargs = {
            "circ": circ,
            "run_fn": run_fn,
            "grad_fn": grad_fn,
            "estimator": estimator,
        }

        if method == "external":
            return ExecuteCircuits.apply(parameters, kwargs)
        if method == "internal":
            from ..ansatz import QuantumNeuralNetwork

            assert isinstance(circ, QuantumNeuralNetwork)
            return ExecuteCircuits.apply(circ.weights, kwargs)
        raise NotImplementedError(f"Unsupported execution method: {method}")


class ExecuteCircuits(torch.autograd.Function):
    """Parameters are input from previous layers"""

    @staticmethod
    def forward(ctx, parameters, kwargs):
        ctx.run_fn = kwargs["run_fn"]
        ctx.circ = kwargs["circ"]
        ctx.estimator = kwargs["estimator"]
        ctx.save_for_backward(parameters)
        parameters = parameters.numpy().tolist()
        outputs = []
        for para in parameters:
            out = ctx.run_fn(ctx.circ, para, estimator=ctx.estimator)
            outputs.append(out)
        outputs = np.stack(outputs)
        outputs = torch.from_numpy(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_out):
        (parameters,) = ctx.saved_tensors
        jac = jacobian(ctx.circ, parameters.numpy(), estimator=ctx.estimator)
        vjp = compute_vjp(jac, grad_out.numpy())
        vjp = torch.from_numpy(vjp)
        return vjp, None
