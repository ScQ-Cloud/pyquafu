
class Backend(object):
    def __init__(self, name):
        self.name = name

    def get_topo(self):
        pass
        
class ScQ_P10(Backend):
    def __init__(self):
        super().__init__("ScQ-P10")

class ScQ_P20(Backend):
    def __init__(self):
        super().__init__("ScQ-P20")

class ScQ_P50(Backend):
    def __init__(self):
        super().__init__("ScQ-P50")