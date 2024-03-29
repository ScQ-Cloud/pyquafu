import copy
from typing import Union

import _operator
import autograd.numpy as anp
from autograd import grad


class ParameterExpression:
    def __init__(self, pivot: "Parameter", value=0.0, operands=[], funcs=[], latex=""):
        self.pivot = pivot
        self.value = value
        self.operands = operands
        self.funcs = funcs
        self.latex = latex  # TODO:Generate latex source code when apply operation

    @property
    def _variables(self):
        vars = {self.pivot: 0}
        vi = 1
        for o in self.operands:
            if isinstance(o, Parameter):
                if o not in vars.keys():
                    vars[o] = vi
                    vi += 1
            elif isinstance(o, ParameterExpression):
                for v in o._variables:
                    if v not in vars.keys():
                        vars[v] = vi
                        vi += 1
        return vars

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
        vars = self._variables

        def __func(x):
            z = x[0]
            for i in range(len(self.funcs)):
                f = self.funcs[i]
                op = self.operands[i]
                if op is None:
                    z = f(z)
                elif isinstance(op, float) or isinstance(op, int):
                    z = f(z, op)
                elif isinstance(op, Parameter):
                    z = f(z, x[vars[op]])
                elif isinstance(op, ParameterExpression):
                    opvars = op._variables.keys()
                    varind = [vars[ov] for ov in opvars]
                    z = f(z, op._func(x[varind]))
                else:
                    raise NotImplementedError
            return z

        return __func

    def __repr__(self):
        return str(self.get_value())

    def grad(self, input=anp.array([])):
        g = grad(self._func)
        if len(input) == 0:
            input = anp.array([v.value for v in self._variables])
        gd = g(input)
        return gd

    def get_value(self):
        input = anp.array([v.value for v in self._variables])
        value = self._func(input)
        self.value = value
        return value

    def __add__(self, r):
        operands = [opr for opr in self.operands]
        funcs = copy.deepcopy(self.funcs)
        operands.append(r)
        funcs.append(_operator.add)
        v = 0.0
        if isinstance(r, Parameter) or isinstance(r, ParameterExpression):
            v = self.value + r.value
        elif isinstance(r, float) or isinstance(r, int):
            v = self.value + r
        else:
            raise NotImplementedError

        return ParameterExpression(self.pivot, v, operands, funcs)

    def __radd__(self, r):
        return self + r

    def __mul__(self, r):
        operands = [opr for opr in self.operands]
        funcs = copy.deepcopy(self.funcs)
        operands.append(r)
        funcs.append(_operator.mul)
        v = 0.0
        if isinstance(r, Parameter) or isinstance(r, ParameterExpression):
            v = self.value * r.value
        elif isinstance(r, float) or isinstance(r, int):
            v = self.value * r
        else:
            raise NotImplementedError

        return ParameterExpression(self.pivot, v, operands, funcs)

    def __rmul__(self, r):
        return self * r

    def __neg__(self):
        return -1.0 * self

    def __sub__(self, r):
        operands = [opr for opr in self.operands]
        funcs = copy.deepcopy(self.funcs)
        operands.append(r)
        funcs.append(_operator.sub)
        v = 0.0
        if isinstance(r, Parameter) or isinstance(r, ParameterExpression):
            v = self.value - r.value
        elif isinstance(r, float) or isinstance(r, int):
            v = self.value - r
        else:
            raise NotImplementedError

        return ParameterExpression(self.pivot, v, operands, funcs)

    def __rsub__(self, r):
        return r - self

    def __truediv__(self, r):
        operands = [opr for opr in self.operands]
        funcs = copy.deepcopy(self.funcs)
        operands.append(r)
        funcs.append(_operator.truediv)
        v = 0.0
        if isinstance(r, Parameter) or isinstance(r, ParameterExpression):
            v = self.value / r.value
        elif isinstance(r, float) or isinstance(r, int):
            v = self.value / r
        else:
            raise NotImplementedError

        return ParameterExpression(self.pivot, v, operands, funcs)

    def __rtruediv__(self, r):
        return r / self

    def __pow__(self, n):
        operands = [opr for opr in self.operands]
        funcs = copy.deepcopy(self.funcs)
        operands.append(n)
        funcs.append(_operator.pow)
        v = 0.0
        if isinstance(n, float) or isinstance(n, int):
            v = self.value**n
        else:
            raise NotImplementedError
        return ParameterExpression(self.pivot, v, operands, funcs)

    def sin(self):
        operands = [opr for opr in self.operands]
        funcs = copy.deepcopy(self.funcs)
        operands.append(None)
        funcs.append(anp.sin)
        v = anp.sin(self.value)
        return ParameterExpression(self.pivot, v, operands, funcs)

    def cos(self):
        operands = [opr for opr in self.operands]
        funcs = copy.deepcopy(self.funcs)
        operands.append(None)
        funcs.append(anp.cos)
        v = anp.cos(self.value)
        return ParameterExpression(self.pivot, v, operands, funcs)

    def tan(self):
        operands = [opr for opr in self.operands]
        funcs = copy.deepcopy(self.funcs)
        operands.append(None)
        funcs.append(anp.tan)
        v = anp.tan(self.value)
        return ParameterExpression(self.pivot, v, operands, funcs)

    def arcsin(self):
        operands = [opr for opr in self.operands]
        funcs = copy.deepcopy(self.funcs)
        operands.append(None)
        funcs.append(anp.arcsin)
        v = anp.arcsin(self.value)
        return ParameterExpression(self.pivot, v, operands, funcs)

    def arccos(self):
        operands = [opr for opr in self.operands]
        funcs = copy.deepcopy(self.funcs)
        operands.append(None)
        funcs.append(anp.arccos)
        v = anp.arccos(self.value)
        return ParameterExpression(self.pivot, v, operands, funcs)

    def arctan(self):
        operands = [opr for opr in self.operands]
        funcs = copy.deepcopy(self.funcs)
        operands.append(None)
        funcs.append(anp.arctan)
        v = anp.arctan(self.value)
        return ParameterExpression(self.pivot, v, operands, funcs)

    def exp(self):
        operands = [opr for opr in self.operands]
        funcs = copy.deepcopy(self.funcs)
        operands.append(None)
        funcs.append(anp.exp)
        v = anp.exp(self.value)
        return ParameterExpression(self.pivot, v, operands, funcs)

    def log(self):
        operands = [opr for opr in self.operands]
        funcs = copy.deepcopy(self.funcs)
        operands.append(None)
        funcs.append(anp.log)
        v = anp.log(self.value)
        return ParameterExpression(self.pivot, v, operands, funcs)


class Parameter(ParameterExpression):
    def __init__(self, name, value: float = 0.0):
        self.name = name
        self.value = float(value)
        self.operands = []
        self.funcs = []
        self.latex = self.name

    @property
    def pivot(self):
        return self

    def grad(self):
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
