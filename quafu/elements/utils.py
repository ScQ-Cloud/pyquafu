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


from typing import Iterable, List, Union
import numpy as np
from quafu.elements.parameters import ParameterType, Parameter, ParameterExpression
import _operator
import autograd.numpy as anp

def reorder_matrix(matrix: np.ndarray, pos: List):
    """Reorder the input sorted matrix to the pos order"""
    qnum = len(pos)
    dim = 2**qnum
    inds = np.argsort(pos)
    inds = np.concatenate([inds, inds + qnum])
    tensorm = np.reshape(matrix, [2] * 2 * qnum)
    return np.transpose(tensorm, inds).reshape([dim, dim])


def extract_float(paras):
    if not isinstance(paras, Iterable):
        paras = [paras]
    paras_f = []
    for para in paras:
        if isinstance(para, float) or isinstance(para, int):
            paras_f.append(para)
        elif isinstance(para, Parameter) or isinstance(para, ParameterExpression):
            paras_f.append(para.get_value())
    return paras_f

def handle_expression(param: ParameterType):
    if isinstance(param, float) or isinstance(param, int):
        return param
    if param.latex:
        return param.latex
    retstr = handle_expression(param.pivot)
    for i in range(len(param.funcs)):
        if param.funcs[i] == _operator.add:
            retstr = f"({retstr} + {handle_expression(param.operands[i])})"
        elif param.funcs[i] == _operator.mul:
            retstr = f"{retstr} * {handle_expression(param.operands[i])}"
        elif param.funcs[i] == _operator.sub:
            retstr = f"({retstr} - {handle_expression(param.operands[i])})"
        elif param.funcs[i] == _operator.truediv:
            retstr = f"{retstr} / {handle_expression(param.operands[i])}"
        elif param.funcs[i] == _operator.pow:
            retstr = f"({retstr}) ^ {handle_expression(param.operands[i])}"
        elif param.funcs[i] == anp.sin:
            retstr = f"sin({retstr})"
        elif param.funcs[i] == anp.cos:
            retstr = f"cos({retstr})"
        elif param.funcs[i] == anp.tan:
            retstr = f"tan({retstr})"
        elif param.funcs[i] == anp.arcsin:
            retstr = f"asin({retstr})"
        elif param.funcs[i] == anp.arccos:
            retstr = f"acos({retstr})"
        elif param.funcs[i] == anp.arctan:
            retstr = f"atan({retstr})"
        elif param.funcs[i] == anp.exp:
            retstr = f"exp({retstr})"
        elif param.funcs[i] == anp.log:
            retstr = f"ln({retstr})"
    return retstr
