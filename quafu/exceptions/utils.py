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

from .circuit_error import IndexOutOfRangeError, InvalidParaError, UnsupportedYet
from .quafu_error import CircuitError, CompileError, QuafuError, ServerError
from .user_error import APITokenNotFound, BackendNotAvailable, UserError


def validate_server_resp(res):
    """Check results returned by backend service"""

    status_code = res["status"] if "status" in res else res["code"]
    err_msg = (
        res["message"]
        if "message" in res
        else res["msg"]
        if "msg" in res
        else "No error message"
    )

    if status_code in [201, 205, 400]:
        raise UserError(err_msg)
    if status_code == 5001:
        raise CircuitError(err_msg)
    if status_code == 5003:
        raise ServerError(err_msg)
    if status_code == 5004:
        raise CompileError(err_msg)
