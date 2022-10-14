from ..utils.platform import get_homedir
import os
import requests
import json
from urllib import parse

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
    
