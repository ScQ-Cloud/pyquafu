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
"""Utils."""
# pylint: disable=no-member

from typing import Iterable, List

import _operator
import autograd.numpy as anp  # pylint: disable=import-error
import numpy as np
from quafu.elements.parameters import Parameter, ParameterExpression, ParameterType


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
        if isinstance(para, (float, int)):
            paras_f.append(para)
        elif isinstance(para, (Parameter, ParameterExpression)):
            paras_f.append(para.get_value())
    return paras_f


# pylint: disable=too-many-branches, too-many-return-statements
def handle_expression(param: ParameterType):
    if isinstance(param, (float, int)):
        return param
    if param.latex:
        return param.latex
    retstr = handle_expression(param.pivot)
    for i, func in enumerate(param.funcs):
        if func == _operator.add:  # pylint: disable=comparison-with-callable
            return f"({retstr} + {handle_expression(param.operands[i])})"
        if func == _operator.mul:  # pylint: disable=comparison-with-callable
            return f"{retstr} * {handle_expression(param.operands[i])}"
        if func == _operator.sub:  # pylint: disable=comparison-with-callable
            return f"({retstr} - {handle_expression(param.operands[i])})"
        if func == _operator.truediv:  # pylint: disable=comparison-with-callable
            return f"{retstr} / {handle_expression(param.operands[i])}"
        if func == _operator.pow:  # pylint: disable=comparison-with-callable
            return f"({retstr}) ^ {handle_expression(param.operands[i])}"
        if func == anp.sin:
            return f"sin({retstr})"
        if func == anp.cos:
            return f"cos({retstr})"
        if func == anp.tan:
            return f"tan({retstr})"
        if func == anp.arcsin:
            return f"asin({retstr})"
        if func == anp.arccos:
            return f"acos({retstr})"
        if func == anp.arctan:
            return f"atan({retstr})"
        if func == anp.exp:
            return f"exp({retstr})"
        if func == anp.log:
            return f"ln({retstr})"
    return retstr
