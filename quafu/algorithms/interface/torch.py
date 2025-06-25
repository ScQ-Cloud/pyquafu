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
# pylint: disable=abstract-method
"""Quafu PyTorch quantum layer."""

from typing import Optional

import numpy as np
import torch
from torch import nn

from ...circuits import QuantumCircuit
from ..ansatz import QuantumNeuralNetwork
from ..estimator import Estimator
from ..gradients import compute_vjp, jacobian, run_circ


# TODO(zhaoyilun): impl a ABC for transformers
class TorchTransformer:
    @staticmethod
    def init_weights(shape):
        """Return torch gradient tensor with specified shape"""
        return torch.randn(*shape, requires_grad=True, dtype=torch.double)

    # TODO(zhaoyilun): docstrings
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    @staticmethod
    def execute(
        circ: QuantumCircuit,
        parameters: torch.Tensor,
        run_fn=run_circ,
        grad_fn=None,
        method="internal",  # pylint: disable=unused-argument
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

        return ExecuteCircuits.apply(parameters, kwargs)


class ExecuteCircuits(torch.autograd.Function):
    """Parameters are input from previous layers"""

    @staticmethod
    def forward(ctx, parameters, kwargs):  # pylint: disable=arguments-differ
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
    def backward(ctx, grad_out):  # pylint: disable=arguments-differ
        (parameters,) = ctx.saved_tensors
        jac = jacobian(ctx.circ, parameters.numpy(), estimator=ctx.estimator)
        vjp = compute_vjp(jac, grad_out.numpy())
        vjp = torch.from_numpy(vjp)
        return vjp, None


class ModuleWrapper(nn.Module):
    """
    A wrapper class to transform quafu circuit to a torch module
    """

    def __init__(self, qnn: QuantumNeuralNetwork):
        """
        Initialization of quafu torch module

        Args:
            circ (QuantumCircuit): the original parameterized quantum circuit
        """
        super().__init__()
        self._qnn = qnn
        if qnn.weights is not None:
            self.weights = nn.parameter.Parameter(qnn.weights)
        else:
            self.weights = None

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (torch.Tensor): raw input data or output from previous
                classical/quantum layers.
        """
        # if weights are not empty, it will be combined with inputs to form
        # the complete parameter vector and feed to the quantum circuit
        bsz, _ = inputs.shape  # FIXME: currently we assume 2-D inputs

        # use the last dimension since it is currently initialized as (1, D)
        if self.weights is not None:
            weight_dim = self.weights.size(-1)
            weights_expanded = self.weights.expand(bsz, weight_dim)
            inputs_to_circ = torch.cat((inputs, weights_expanded), dim=1)
        else:
            inputs_to_circ = inputs
        return self._qnn(inputs_to_circ)
