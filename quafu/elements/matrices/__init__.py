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
"""Matrices module."""

from .mat_lib import (
    CTMatrix,
    CXMatrix,
    CYMatrix,
    CZMatrix,
    FredkinMatrix,
    HMatrix,
    IdMatrix,
    ISwapMatrix,
    SMatrix,
    SwapMatrix,
    SWMatrix,
    SXMatrix,
    SYMatrix,
    TMatrix,
    ToffoliMatrix,
    WMatrix,
    XMatrix,
    YMatrix,
    ZMatrix,
    mat_dict,
    pmatrix,
    rx_mat,
    rxx_mat,
    ry_mat,
    ryy_mat,
    rz_mat,
    rzz_mat,
    u2matrix,
    u3matrix,
)
from .mat_utils import is_hermitian, is_zero, stack_matrices

__all__ = [
    "IdMatrix",
    "XMatrix",
    "YMatrix",
    "ZMatrix",
    "SMatrix",
    "SXMatrix",
    "SYMatrix",
    "TMatrix",
    "WMatrix",
    "SWMatrix",
    "HMatrix",
    "SwapMatrix",
    "ISwapMatrix",
    "CXMatrix",
    "CYMatrix",
    "CZMatrix",
    "CTMatrix",
    "ToffoliMatrix",
    "FredkinMatrix",
    "rx_mat",
    "ry_mat",
    "rz_mat",
    "pmatrix",
    "rxx_mat",
    "ryy_mat",
    "rzz_mat",
    "u2matrix",
    "u3matrix",
    "mat_dict",
    "is_hermitian",
    "is_zero",
    "stack_matrices",
]
