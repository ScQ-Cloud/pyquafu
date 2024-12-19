# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Test quafu dtype."""

import quafu
from quafu.simulator import Simulator
import numpy as np
import pytest


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize(
    "error_dtype",
    [
        'int',
        'float',
        'quafu.int',
        'quafu.flaot',
        'quafu.double',
        'quafu.complex',
        'quafu.int32',
        'quafu.float31',
        'quafu.float63',
        'quafu.complex63',
        'quafu.complex127',
        'np.float32',
    ],
)
def test_error_dtype(error_dtype):
    """
    Description: test error dtype.
    Expectation: raise error
    """
    with pytest.raises(ValueError):
        Simulator('quafuvector', 2, dtype=error_dtype)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dtype():
    """
    Description: test quafu dtype.
    Expectation: success
    """
    assert quafu.dtype.is_double_precision(quafu.float64)
    assert quafu.dtype.is_double_precision(quafu.complex128)
    assert not quafu.dtype.is_double_precision(quafu.float32)
    assert not quafu.dtype.is_double_precision(quafu.complex64)

    assert quafu.dtype.is_single_precision(quafu.float32)
    assert quafu.dtype.is_single_precision(quafu.complex64)
    assert not quafu.dtype.is_single_precision(quafu.float64)
    assert not quafu.dtype.is_single_precision(quafu.complex128)

    assert quafu.dtype.is_same_precision(quafu.float32, quafu.complex64)
    assert quafu.dtype.is_same_precision(quafu.float64, quafu.complex128)
    assert not quafu.dtype.is_same_precision(quafu.float32, quafu.complex128)

    assert quafu.dtype.precision_str(quafu.float32) == 'single precision'
    assert quafu.dtype.precision_str(quafu.float64) == 'double precision'
    assert not quafu.dtype.precision_str(quafu.complex64) == 'double precision'

    assert quafu.dtype.to_real_type(quafu.complex64) == quafu.float32
    assert quafu.dtype.to_real_type(quafu.complex128) == quafu.float64
    assert quafu.dtype.to_real_type(quafu.float64) == quafu.float64

    assert quafu.dtype.to_complex_type(quafu.float32) == quafu.complex64
    assert quafu.dtype.to_complex_type(quafu.float64) == quafu.complex128
    assert quafu.dtype.to_complex_type(quafu.complex128) == quafu.complex128

    assert quafu.dtype.to_single_precision(quafu.float64) == quafu.float32
    assert quafu.dtype.to_single_precision(quafu.complex128) == quafu.complex64
    assert quafu.dtype.to_single_precision(quafu.float32) == quafu.float32

    assert quafu.dtype.to_double_precision(quafu.float32) == quafu.float64
    assert quafu.dtype.to_double_precision(quafu.complex64) == quafu.complex128
    assert quafu.dtype.to_double_precision(quafu.float32) == quafu.float64

    assert quafu.dtype.to_precision_like(quafu.float32, quafu.complex128) == quafu.float64
    assert quafu.dtype.to_precision_like(quafu.complex128, quafu.float32) == quafu.complex64

    assert quafu.dtype.to_quafu_type(np.float32) == quafu.float32
    assert quafu.dtype.to_quafu_type(np.float64) == quafu.float64
    assert quafu.dtype.to_quafu_type(np.complex64) == quafu.complex64
    assert quafu.dtype.to_quafu_type(np.complex128) == quafu.complex128

    assert quafu.dtype.to_np_type(quafu.float32) == np.float32
    assert quafu.dtype.to_np_type(quafu.complex128) == np.complex128
