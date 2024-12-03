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
"""Exceptions for errors raised while building circuit."""


class QuafuError(Exception):
    """Base class for errors raised by Quafu."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(" ".join(message))
        self.message = " ".join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)


class CircuitError(QuafuError):
    """Exceptions for errors raised while building circuit."""


class ServerError(QuafuError):
    """Exceptions for errors raised while connecting to server."""


class CompileError(QuafuError):
    """Exceptions for errors raised while compiling circuit."""
