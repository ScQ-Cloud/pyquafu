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

from typing import Any
import torch
from quafu import QuantumCircuit
from quafu import QuantumCircuit, simulate
from quafu.elements.element_gates.matrices import ZMatrix


class ExecuteCircuits(torch.autograd.Function):
    """TODO(zhaoyilun): document"""

    @staticmethod
    def forward(ctx, parameters, **kwargs) -> Any:
        ctx.run_fn = kwargs["run_fn"]
        ctx.circ = kwargs["circ"]
        out = ctx.run_fn(ctx.circ, parameters)
        return out

    @staticmethod
    def backward(ctx: Any, grad_out) -> Any:
        circ, grad_fn = ctx.saved_tensors
        grad = grad_fn(circ, grad_out)
        return grad


# TODO(zhaoyilun): doc
def execute(circ: QuantumCircuit, run_fn, grad_fn, parameters: torch.Tensor):
    """execute.

    Args:
        circ:
        run_fn:
        grad_fn:
    """

    kwargs = {"circ": circ, "run_fn": run_fn, "grad_fn": grad_fn}

    return ExecuteCircuits.apply(parameters, **kwargs)
