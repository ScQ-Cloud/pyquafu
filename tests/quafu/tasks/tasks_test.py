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

import json
import unittest
from unittest.mock import patch

from quafu.backends.backends import Backend
from quafu.exceptions import ServerError
from quafu.exceptions.quafu_error import CircuitError, CompileError
from quafu.exceptions.user_error import UserError

from quafu import QuantumCircuit, Task, User

DUMMY_API_TOKEN = "123456"

DUMMY_BACKENDS = {}
with open("tests/quafu/tasks/data/fake_backends.json", "r") as f:
    resp = json.loads(f.read())
    DUMMY_BACKENDS = {info["system_name"]: Backend(info) for info in resp["data"]}


DUMMY_TASK_RES_DATA = {}
with open("tests/quafu/tasks/data/fake_task_res.json", "r") as f:
    DUMMY_TASK_RES_DATA = json.loads(f.read())


DUMMY_CIRC = QuantumCircuit(2)
DUMMY_CIRC.h(0)
DUMMY_CIRC.cx(0, 1)
DUMMY_CIRC.measure()


class MockFailedResponse:
    def __init__(self, status: int, msg: str) -> None:
        self.ok = True
        self.status_code = 200
        self._status = status
        self._msg = msg

    def json(self):
        return {"status": self._status, "msg": self._msg}


# FIXME: maybe no need to use this
class MockSucceededResponse:
    def __init__(self) -> None:
        self.ok = True
        self.status_code = 200

    def json(self):
        return DUMMY_TASK_RES_DATA


DUMMY_TASK_RES_FAILED_USER = MockFailedResponse(400, "Dummy user error")
DUMMY_TASK_RES_FAILED_CIRCUIT = MockFailedResponse(5001, "Dummy circuit error")
DUMMY_TASK_RES_FAILED_SERVER = MockFailedResponse(5003, "Dummy server error")
DUMMY_TASK_RES_FAILED_COMPILE = MockFailedResponse(5004, "Dummy compile error")


class TestTask(unittest.TestCase):
    @patch("quafu.users.userapi.User.get_available_backends")
    @patch("quafu.utils.client_wrapper.ClientWrapper.post")
    def test_send(self, mock_post, mock_get_available_backends):
        mock_get_available_backends.return_value = DUMMY_BACKENDS
        user = User(api_token=DUMMY_API_TOKEN)
        task = Task(user=user)

        # 1. Requests library throws exception
        mock_post.side_effect = ServerError()
        with self.assertRaises(ServerError):
            task.send(DUMMY_CIRC)

        # 2. Website service set customized status code for some errors
        mock_post.side_effect = None

        mock_post.return_value = DUMMY_TASK_RES_FAILED_USER
        with self.assertRaisesRegex(UserError, "Dummy user error"):
            task.send(DUMMY_CIRC)

        mock_post.return_value = DUMMY_TASK_RES_FAILED_CIRCUIT
        with self.assertRaisesRegex(CircuitError, "Dummy circuit error"):
            task.send(DUMMY_CIRC)

        mock_post.return_value = DUMMY_TASK_RES_FAILED_SERVER
        with self.assertRaisesRegex(ServerError, "Dummy server error"):
            task.send(DUMMY_CIRC)

        mock_post.return_value = DUMMY_TASK_RES_FAILED_COMPILE
        with self.assertRaisesRegex(CompileError, "Dummy compile error"):
            task.send(DUMMY_CIRC)
