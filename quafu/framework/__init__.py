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
"""Quantum neural networks operators and cells."""
import warnings

framework_modules = [
    "QUAFUAnsatzOnlyLayer",
    "QUAFUN2AnsatzOnlyLayer",
    "QUAFULayer",
    "QUAFUN2Layer",
    "QUAFUOps",
    "QUAFUN2Ops",
    "QUAFUAnsatzOnlyOps",
    "QUAFUN2AnsatzOnlyOps",
    "QUAFUEncoderOnlyOps",
    "QUAFUN2EncoderOnlyOps",
    "QRamVecOps",
    "QRamVecLayer",
]

__all__ = []
try:
    import mindspore

    from .layer import (
        QUAFUAnsatzOnlyLayer,
        QUAFULayer,
        QUAFUN2AnsatzOnlyLayer,
        QUAFUN2Layer,
        QRamVecLayer,
    )
    from .operations import (
        QUAFUAnsatzOnlyOps,
        QUAFUEncoderOnlyOps,
        QUAFUN2AnsatzOnlyOps,
        QUAFUN2EncoderOnlyOps,
        QUAFUN2Ops,
        QUAFUOps,
        QRamVecOps,
    )

    __all__.extend(framework_modules)
    import packaging.version

    ms_version = mindspore.__version__
    if "rc" in ms_version:
        ms_version = ms_version[: ms_version.find('rc')]
    ms_requires = packaging.version.parse('1.4.0')
    if packaging.version.parse(ms_version) < ms_requires:
        warnings.warn(
            "Current version of MindSpore is not compatible with MindSpore Quantum. "
            "Some functions might not work or even raise error. Please install MindSpore "
            "version >= 1.4.0. For more details about dependency setting, please check "
            "the instructions at MindSpore official website https://www.mindspore.cn/install "
            "or check the README.md at https://gitee.com/mindspore/quafu",
            stacklevel=2,
        )

except ImportError:
    pass

__all__.sort()


# pylint: disable=invalid-name
def __getattr__(name):
    if name in framework_modules:
        raise ImportError(
            f"cannot import '{name}' from 'quafu.framework'. "
            "MindSpore not installed, 'quafu.framework' modules "
            "(for hybrid quantum-classical neural network) are disabled."
        )
    raise ImportError(f"cannot import '{name}' from 'quafu'. '{name}' does not exist in quafu.")
