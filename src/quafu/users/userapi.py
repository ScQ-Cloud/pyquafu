import json
import os

import requests

from .exceptions import UserError, APITokenNotFound
from ..utils.platform import get_homedir


class User(object):
    url = "https://quafu.baqis.ac.cn/"
    backends_api = url + "qbackend/get_backends/"
    chip_api = url + "qbackend/scq_get_chip_info/"
    exec_api = url + "qbackend/scq_kit/"
    exec_async_api = url + "qbackend/scq_kit_asyc/"
    exec_recall_api = url + "qbackend/scq_task_recall/"

    def __init__(self, api_token: str = None, token_dir: str = None):
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
            self._api_token = self._load_account_token()
        else:
            self._api_token = api_token

        self.priority = 2

    @property
    def api_token(self):
        if self._api_token is None:
            raise APITokenNotFound(f"API token not set, neither found at dir: '{self.token_dir}'. "
                                   "Please set up by providing api_token/token_dir when initializing User.")
        return self._api_token

    def save_apitoken(self, apitoken=None):
        """
        Save api-token associate your Quafu account.
        """
        if apitoken is not None:
            import warnings
            warnings.warn("The argument 'apitoken' in this function will be deprecated "
                          "in the future, please set api token by providing 'api_token' "
                          "or 'token_dir' when initialize User()."
                          )
            self._api_token = apitoken

        file_dir = self.token_dir
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        with open(file_dir + "api", "w") as f:
            f.write(self.api_token + "\n")
            f.write(self.url)

    def _load_account_token(self):
        """
        Load Quafu account, only api at present.

        TODO: expand to load more user information
        """
        file_dir = self.token_dir + "api"
        if not os.path.exists(file_dir):
            return None
        else:
            f = open(file_dir, "r")
            token = f.readline().strip()
        return token

    def _get_backends_info(self):
        """
        Get available backends information
        """

        backends_info = requests.post(url=self.backends_api, headers={"api_token": self.api_token})
        backends_info_dict = json.loads(backends_info.text)
        if backends_info_dict["status"] == 201:
            raise UserError(backends_info_dict["message"])
        else:
            return backends_info_dict["data"]

    def get_available_backends(self, print_info=True):
        """
        Get available backends
        """
        from quafu.backends.backends import Backend
        backends_info = self._get_backends_info()
        self._available_backends = {info["system_name"]: Backend(info) for info in backends_info}

        if print_info:
            print((" " * 5).join(["system_name".ljust(10), "qubits".ljust(10), "status".ljust(10)]))
            for backend in self._available_backends.values():
                print((" " * 5).join(
                    [backend.name.ljust(10), str(backend.qubit_num).ljust(10), backend.status.ljust(10)]))

        return self._available_backends
