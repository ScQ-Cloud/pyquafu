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

from .parameters import Parameter, ParameterExpression


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
