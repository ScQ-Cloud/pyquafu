from ..utils.platform import get_homedir
import os
import requests
import json
from urllib import parse
from .exceptions import UserError

class User(object):
    def __init__(self):
        self.apitoken = ""
        
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
    
def load_account():
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
        return token, url
    except:
        raise UserError("User configure error. Please set up your token.")
    
def get_backends_info():
    """
    Get available backends information
    """
    token, _url = load_account()
    backends_info = requests.post(url=_url+"qbackend/get_backends/", headers={"api_token" : token})
    backends_dict = json.loads(backends_info.text)
    return backends_dict["data"]
