# Copyright 2021 Huawei Technologies Co., Ltd
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

"""QASM flavors support (OpenQASM, HiQASM, etc.)."""

from .hiqasm import HiQASM, random_hiqasm
from .openqasm import OpenQASM
from .qcis import QCIS

__all__ = ['OpenQASM', 'random_hiqasm', 'HiQASM', 'QCIS']

__all__.sort()
