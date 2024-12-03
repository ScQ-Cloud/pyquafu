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
"""Exceptions."""
from .circuit_error import IndexOutOfRangeError, InvalidParaError, UnsupportedYet
from .quafu_error import CircuitError, CompileError, QuafuError, ServerError
from .user_error import (
    APITokenNotFound,
    BackendNotAvailable,
    InvalidAPIToken,
    UserError,
)
from .utils import validate_server_resp

__all__ = [
    "IndexOutOfRangeError",
    "InvalidParaError",
    "UnsupportedYet",
    "QuafuError",
    "CircuitError",
    "ServerError",
    "CompileError",
    "UserError",
    "APITokenNotFound",
    "InvalidAPIToken",
    "BackendNotAvailable",
    "validate_server_resp",
]
