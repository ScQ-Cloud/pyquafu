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
"""List available simulators."""
import typing
import warnings

from quafu import _quafu_matrix, _quafu_vector, quafubackend
from quafu.dtype import complex64, complex128
from quafu.simulator.backend_base import BackendBase
from quafu.utils.error import SimNotAvailableError

GPU_DISABLED_REASON = None

try:
    from quafu import _quafu_vector_gpu

    # pylint: disable=no-member
    _quafu_vector_gpu.double.quafuvector_gpu(1).apply_gate(quafubackend.gate.HGate([0]))
    QUAFUVECTOR_GPU_SUPPORTED = True
except ImportError as err:
    GPU_DISABLED_REASON = f"Unable to import quafuvector_gpu backend. This backend requires CUDA 11 or higher."
    QUAFUVECTOR_GPU_SUPPORTED = False
except RuntimeError as err:
    GPU_DISABLED_REASON = f"Disable quafuvector gpu backend due to: {err}"
    QUAFUVECTOR_GPU_SUPPORTED = False


class _AvailableSimulator:
    """Set available simulator."""

    def __init__(self):
        """Init available simulator obj."""
        self.base_module = {
            'quafuvector': _quafu_vector,
            'quafumatrix': _quafu_matrix,
            'stabilizer': _quafu_vector,
        }
        self.sims = {
            'quafuvector': {
                complex64: _quafu_vector.float,
                complex128: _quafu_vector.double,
            },
            'quafumatrix': {
                complex64: _quafu_matrix.float,
                complex128: _quafu_matrix.double,
            },
            'stabilizer': _quafu_vector.stabilizer,
        }
        if QUAFUVECTOR_GPU_SUPPORTED:
            self.base_module['quafuvector_gpu'] = _quafu_vector_gpu
            self.sims['quafuvector_gpu'] = {
                complex64: _quafu_vector_gpu.float,
                complex128: _quafu_vector_gpu.double,
            }

    def is_available(self, sim: typing.Union[str, BackendBase], dtype) -> bool:
        """Check a simulator with given data type is available or not."""
        if isinstance(sim, BackendBase):
            return True
        if sim == 'stabilizer':
            return True
        if sim in self.sims and dtype in self.sims[sim]:
            return True
        return False

    def c_module(self, sim: str, dtype=None):
        """Get available simulator c module."""
        if sim == 'quafuvector_gpu' and not QUAFUVECTOR_GPU_SUPPORTED:
            warnings.warn(f"{GPU_DISABLED_REASON}", stacklevel=3)
        if dtype is None:
            if sim not in self.base_module:
                raise SimNotAvailableError(sim)
            return self.base_module[sim]
        if not self.is_available(sim, dtype):
            raise SimNotAvailableError(sim, dtype)
        return self.sims[sim][dtype]

    def py_class(self, sim: str):
        """Get python base class of simulator."""
        if sim in self.sims:
            if sim in ['quafuvector', 'quafuvector_gpu', 'quafumatrix']:
                # pylint: disable=import-outside-toplevel
                from quafu.simulator.quafusim import QUAFUSim

                return QUAFUSim
            if sim == 'stabilizer':
                # pylint: disable=import-outside-toplevel
                from quafu.simulator.stabilizer import Stabilizer

                return Stabilizer
            raise SimNotAvailableError(sim)
        raise SimNotAvailableError(sim)

    def __iter__(self):
        """List available simulator with data type."""
        for k, v in self.sims.items():
            if not isinstance(v, dict):
                yield k
            else:
                for dtype in v:
                    yield [k, dtype]


SUPPORTED_SIMULATOR = _AvailableSimulator()
