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
import os
import warnings
from typing import Optional

from ..exceptions import APITokenNotFound, UserError, validate_server_resp
from ..utils.client_wrapper import ClientWrapper
from ..utils.platform import get_homedir


class User(object):
    url = "https://quafu.baqis.ac.cn/"
    backends_api = "qbackend/get_backends/"
    chip_api = "qbackend/scq_get_chip_info/"
    exec_api = "qbackend/scq_kit/"
    exec_async_api = "qbackend/scq_kit_asyc/"
    exec_recall_api = "qbackend/scq_task_recall/"
    backend_type = "quafu"

    def __init__(
        self, api_token: Optional[str] = None, token_dir: Optional[str] = None
    ):
        """
        Initialize user account and load backend information.

        :param api_token: if provided
        :param token_dir: where api token is found or saved
        """
        self._available_backends = {}

        if token_dir is None:
            self.token_dir = get_homedir() + "/.quafu/"
        else:
            self.token_dir = token_dir

        if api_token is None:
            self._api_token = self._load_account()
        else:
            self._api_token = api_token

        self.priority = 2

    @classmethod
    def list_configurable_attributes(cls):
        """Return all attributes that are configuratble in configuration file"""
        # get all configurable attributes
        attributes = [
            attr
            for attr in list(cls.__dict__.keys())
            if not attr.startswith("__")
            and not callable(getattr(cls, attr))
            and not isinstance(getattr(cls, attr), property)
        ]
        return attributes

    @property
    def api_token(self):
        if self._api_token is None:
            raise APITokenNotFound(
                f"API token not set, neither found at dir: '{self.token_dir}'. "
                "Please set up by providing api_token/token_dir when initializing User."
            )
        return self._api_token

    def save_apitoken(self, apitoken=None):
        """
        Save the api-token of your Quafu account.
        """
        if apitoken is not None:
            import warnings

            warnings.warn(
                "The argument 'apitoken' in this function will be deprecated "
                "in the future, please set api token by providing 'api_token' "
                "or 'token_dir' when initialize User()."
            )
            self._api_token = apitoken

        file_dir = self.token_dir
        file_path = os.path.join(file_dir, "api")

        data = None
        if os.path.exists(file_path):
            # if the configuration file exists
            # only update api token
            with open(file_path, "r") as f:
                data = json.load(f)
            data["token"] = self.api_token
        else:
            # if the configuration file does not exist
            # set default configuration for api_token and url
            data = {"token": self.api_token, "url": self.url}
            if not os.path.exists(file_dir):
                os.mkdir(file_dir)

        with open(file_path, "w") as f:
            json.dump(data, f)

    def _load_account(self):
        """
        Load Quafu account, if any configurable attribute present in
        configuration file, we will override it
        """
        file_dir = os.path.join(self.token_dir, "api")
        attr_names = self.__class__.list_configurable_attributes()

        if not os.path.exists(file_dir):
            raise UserError("Please first save api token using `User.save_apitoken()`")
        with open(file_dir, "r") as f:
            try:
                data = json.load(f)
                token = data["token"]
                for attr_name in attr_names:
                    if attr_name in data:
                        setattr(self.__class__, attr_name, data[attr_name])
            except json.JSONDecodeError:
                warnings.warn(
                    "Deprecated config file format, suggest to fix this warning by running save_apitoken() again"
                )
                f.seek(0)
                items = f.readlines()
                token = items[0].strip()
                self.__class__.url = items[1].strip()
        return token

    def _get_backends_info(self):
        """
        Get available backends information
        """
        headers = {"api_token": self.api_token}
        url = self.url + self.backends_api
        response = ClientWrapper.post(url=url, headers=headers)
        backends_info = response.json()
        validate_server_resp(backends_info)
        return backends_info["data"]

    def get_available_backends(self, print_info=True):
        """
        Get available backends
        """
        from quafu.backends.backends import Backend

        backends_info = self._get_backends_info()
        self._available_backends = {
            info["system_name"]: Backend(info) for info in backends_info
        }

        if print_info:
            print("\t ".join(["system_name".ljust(10), "qubits".ljust(5), "status"]))
            for backend in self._available_backends.values():
                print(
                    "\t ".join(
                        [
                            backend.name.ljust(10),
                            str(backend.qubit_num).ljust(5),
                            backend.status,
                        ]
                    )
                )
        return self._available_backends
