from ..utils.platform import get_homedir
import os
import requests
import json
from urllib import parse
from .exceptions import UserError


class User(object):
    def __init__(self):
        self.apitoken = ""
        self._url = "http://quafu.baqis.ac.cn/"
        
        if self.check_account_data():
            self.load_account()
            
        self.backends_api = "qbackend/get_backends/"
        self.chip_api = "qbackend/scq_get_chip_info/"
        self.exec_api = "qbackend/scq_kit/"
        self.exec_async_api = "qbackend/scq_kit_asyc/"
        self.exec_recall_api = "qbackend/scq_task_recall/"
        self.priority = 2

    def check_account_data(self):
        homedir = get_homedir()
        file_dir = homedir + "/.quafu/"
        if not os.path.exists(file_dir):
            print("Your user information is not configured. Remember to configure it by User.save_apitoken(<your api_token>)")
            return False
        return True
        # TODO: Check if the user's basic data file is formated

    def save_apitoken(self, apitoken):
        """
        Save your apitoken associate your Quafu account.
        """
        self.apitoken = apitoken
        homedir = get_homedir()
        file_dir = homedir + "/.quafu/"
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        with open(file_dir + "api", "w") as f:
            f.write(self.apitoken+"\n")
            f.write("http://quafu.baqis.ac.cn/")

    def load_account(self):
        """
        Load Quafu account.
        """
        homedir = get_homedir()
        file_dir = homedir + "/.quafu/"
        try:
            f = open(file_dir + "api", "r")
            data = f.readlines()
            token = data[0].strip("\n")
            url = data[1].strip("\n")
            self.apitoken = token
            self._url = url
            return token, url
        except:
            raise UserError("User configure error. Please set up your token.")

    def get_backends_info(self):
        """
        Get available backends information
        """

        backends_info = requests.post(url=self._url+self.backends_api, headers={"api_token" : self.apitoken})
        backends_info_dict = json.loads(backends_info.text)
        if backends_info_dict["status"] == 201:
            raise UserError(backends_info_dict["message"])
        else:
            return backends_info_dict["data"]

    def get_available_backends(self, print_info=True):
        from quafu.backends.backends import Backend
        backends_info = self.get_backends_info()
        self._available_backends = {info["system_name"]:Backend(info) for info in backends_info}

        if print_info:
            print((" "*5).join(["system_name".ljust(10), "qubits".ljust(10),   "status".ljust(10)]))
            for backend in self._available_backends.values():
                print((" "*5).join([backend.name.ljust(10), str(backend.qubit_num).ljust(10),  backend.status.ljust(10)]))

        return self._available_backends
