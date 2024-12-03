# (C) Copyright 2024 Beijing Academy of Quantum Information Sciences
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
"""Parameter module."""
# pylint: disable=no-member
import copy
from typing import Union

import _operator
import autograd.numpy as anp
from autograd import grad


class ParameterExpression:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(self, pivot: "Parameter", value=0.0, operands=None, funcs=None, latex=""):
        if operands is None:
            operands = []
        if funcs is None:
            funcs = []
        self.pivot = pivot
        self.value = value
        self.operands = operands
        self.funcs = funcs
        self.latex = latex  # TODO:Generate latex source code when apply operation

    @property
    def _variables(self):
        variables = {self.pivot: 0}
        vi = 1
        for o in self.operands:
            if isinstance(o, Parameter):
                if o not in variables:
                    variables[o] = vi
                    vi += 1
            elif isinstance(o, ParameterExpression):
                for v in o._variables:
                    if v not in variables:
                        variables[v] = vi
                        vi += 1
        return variables

    def _undo(self, step):
        for _ in range(step):
            if len(self.operands) > 0:
                self.operands.pop()
                self.funcs.pop()
            else:
                return self.pivot
        return self

    @property
    def _func(self):
        variables = self._variables

        def __func(x):
            z = x[0]
            for i, f in enumerate(self.funcs):
                op = self.operands[i]
                if op is None:
                    z = f(z)
                elif isinstance(op, (float, int)):
                    z = f(z, op)
                elif isinstance(op, Parameter):
                    z = f(z, x[variables[op]])
                elif isinstance(op, ParameterExpression):
                    opvars = op._variables.keys()
                    varind = [variables[ov] for ov in opvars]
                    z = f(z, op._func(x[varind]))
                else:
                    raise NotImplementedError
            return z

        return __func

    def __repr__(self):
        return str(self.get_value())

    def grad(self, input_value=anp.array([])):
        g = grad(self._func)  # pylint: disable=no-value-for-parameter
        if len(input_value) == 0:
            input_value = anp.array([v.value for v in self._variables])
        return g(input_value)

    def get_value(self):
        input_value = anp.array([v.value for v in self._variables])
        value = self._func(input_value)
        self.value = value
        return value

    def __add__(self, r):
        operands = list(self.operands)
        funcs = copy.deepcopy(self.funcs)
        operands.append(r)
        funcs.append(_operator.add)
        v = 0.0
        if isinstance(r, (Parameter, ParameterExpression)):
            v = self.value + r.value
        elif isinstance(r, (float, int)):
            v = self.value + r
        else:
            raise NotImplementedError

        return ParameterExpression(self.pivot, v, operands, funcs)

    def __radd__(self, r):
        return self + r

    def __mul__(self, r):
        operands = list(self.operands)
        funcs = copy.deepcopy(self.funcs)
        operands.append(r)
        funcs.append(_operator.mul)
        v = 0.0
        if isinstance(r, (Parameter, ParameterExpression)):
            v = self.value * r.value
        elif isinstance(r, (float, int)):
            v = self.value * r
        else:
            raise NotImplementedError

        return ParameterExpression(self.pivot, v, operands, funcs)

    def __rmul__(self, r):
        return self * r

    def __neg__(self):
        return -1.0 * self

    def __sub__(self, r):
        operands = list(self.operands)
        funcs = copy.deepcopy(self.funcs)
        operands.append(r)
        funcs.append(_operator.sub)
        v = 0.0
        if isinstance(r, (Parameter, ParameterExpression)):
            v = self.value - r.value
        elif isinstance(r, (float, int)):
            v = self.value - r
        else:
            raise NotImplementedError

        return ParameterExpression(self.pivot, v, operands, funcs)

    def __rsub__(self, r):
        return r - self

    def __truediv__(self, r):
        operands = list(self.operands)
        funcs = copy.deepcopy(self.funcs)
        operands.append(r)
        funcs.append(_operator.truediv)
        v = 0.0
        if isinstance(r, (Parameter, ParameterExpression)):
            v = self.value / r.value
        elif isinstance(r, (float, int)):
            v = self.value / r
        else:
            raise NotImplementedError

        return ParameterExpression(self.pivot, v, operands, funcs)

    def __rtruediv__(self, r):
        return r / self

    def __pow__(self, n):
        operands = list(self.operands)
        funcs = copy.deepcopy(self.funcs)
        operands.append(n)
        funcs.append(_operator.pow)
        v = 0.0
        if isinstance(n, (float, int)):
            v = self.value**n
        else:
            raise NotImplementedError
        return ParameterExpression(self.pivot, v, operands, funcs)

    def sin(self):
        operands = list(self.operands)
        funcs = copy.deepcopy(self.funcs)
        operands.append(None)
        funcs.append(anp.sin)
        v = anp.sin(self.value)
        return ParameterExpression(self.pivot, v, operands, funcs)

    def cos(self):
        operands = list(self.operands)
        funcs = copy.deepcopy(self.funcs)
        operands.append(None)
        funcs.append(anp.cos)
        v = anp.cos(self.value)
        return ParameterExpression(self.pivot, v, operands, funcs)

    def tan(self):
        operands = list(self.operands)
        funcs = copy.deepcopy(self.funcs)
        operands.append(None)
        funcs.append(anp.tan)
        v = anp.tan(self.value)
        return ParameterExpression(self.pivot, v, operands, funcs)

    def arcsin(self):
        operands = list(self.operands)
        funcs = copy.deepcopy(self.funcs)
        operands.append(None)
        funcs.append(anp.arcsin)
        v = anp.arcsin(self.value)
        return ParameterExpression(self.pivot, v, operands, funcs)

    def arccos(self):
        operands = list(self.operands)
        funcs = copy.deepcopy(self.funcs)
        operands.append(None)
        funcs.append(anp.arccos)
        v = anp.arccos(self.value)
        return ParameterExpression(self.pivot, v, operands, funcs)

    def arctan(self):
        operands = list(self.operands)
        funcs = copy.deepcopy(self.funcs)
        operands.append(None)
        funcs.append(anp.arctan)
        v = anp.arctan(self.value)
        return ParameterExpression(self.pivot, v, operands, funcs)

    def exp(self):
        operands = list(self.operands)
        funcs = copy.deepcopy(self.funcs)
        operands.append(None)
        funcs.append(anp.exp)
        v = anp.exp(self.value)
        return ParameterExpression(self.pivot, v, operands, funcs)

    def log(self):
        operands = list(self.operands)
        funcs = copy.deepcopy(self.funcs)
        operands.append(None)
        funcs.append(anp.log)
        v = anp.log(self.value)
        return ParameterExpression(self.pivot, v, operands, funcs)


class Parameter(ParameterExpression):
    # pylint: disable=super-init-not-called
    def __init__(self, name, value: float = 0.0, tunable: bool = True):
        self.name = name
        self.value = float(value)
        self.operands = []
        self.funcs = []
        self.latex = self.name
        self.tunable = tunable

    @property
    def pivot(self):
        return self

    def grad(self):  # pylint: disable=arguments-differ
        return anp.array([1.0])

    def get_value(self):
        return self.value

    def __hash__(self):
        return hash((self.name))

    def __eq__(self, other):
        return (self.name) == (other.name)

    def __repr__(self):
        return f"{self.name}({self.value})"


ParameterType = Union[float, Parameter, ParameterExpression]
