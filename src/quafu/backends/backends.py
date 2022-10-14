import requests
import json
import re
import networkx as nx

class Backend(object):
    def __init__(self, name):
        self.name = name
        self.valid_gates = []
        
    def get_info(self, url, api_token):
        data = {"system_name": self.name.lower()}
        headers={"api_token": api_token}
        chip_info = requests.post(url = url + "qbackend/scq_get_chip_info/", data=data, 
        headers=headers)
        backend_info = json.loads(chip_info.text)
       
        return backend_info

    def get_valid_gates(self):
        return self.valid_gates


class ScQ_P10(Backend):
    def __init__(self):
        super().__init__("ScQ-P10")
        self.valid_gates = ["cx", "cz", "rx", "ry", "rz", "x", "y", "z", "h", "sx", "sy", "id", "delay", "barrier", "cy", "cnot", "swap"]

class ScQ_P20(Backend):
    def __init__(self):
        super().__init__("ScQ-P20")
        self.valid_gates = ["cx", "cz", "rx", "ry", "rz", "x", "y", "z", "h", "sx", "sy", "id", "delay", "barrier", "cy", "cnot", "swap"]
        
class ScQ_P50(Backend):
    def __init__(self):
        super().__init__("ScQ-P50")
        self.valid_gates = ["cx", "cz", "rx", "ry", "rz", "x", "y", "z", "h"]

class ScQ_S41(Backend):
    def __init__(self):
        super().__init__("ScQ-S41")
        self.valid_gates = [ "rx", "ry", "rz", "x", "y", "z", "h", "sx", "sy", "id", "delay", "barrier", "xy"]
            