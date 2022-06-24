#This is the file for  concrete element quantum gates
from quantum_element import *

class HGate(SingleQubitGate):
    def __init__(self, pos):
        super().__init__("H", pos)

class XGate(SingleQubitGate):
    def __init__(self,  pos):
        super().__init__("X", pos)

class YGate(SingleQubitGate):
    def __init__(self,  pos):
        super().__init__("Y", pos)

class ZGate(SingleQubitGate):
    def __init__(self,  pos):
        super().__init__("Z", pos)

class RxGate(ParaSingleQubitGate):
    def __init__(self, pos, paras):
        super().__init__("Rx", pos, paras)

class RyGate(ParaSingleQubitGate):
    def __init__(self,  pos, paras):
        super().__init__("Ry", pos, paras)

class RzGate(ParaSingleQubitGate):
    def __init__(self, pos, paras):
        super().__init__("Rz", pos, paras)

class iSWAP(TwoQubitGate):
    def __init__(self, pos):
        super().__init__("iSWAP", pos)

    def to_QLisp(self):
        raise ValueError("The BAQIS backend does not support iSWAP gate currently, please use the IOP backend.")

class CnotGate(ControlGate):
    def __init__(self, ctrl, targ):
        super().__init__("CNOT", ctrl, targ)

class CzGate(ControlGate):
    def __init__(self, ctrl, targ):
        super().__init__("Cz", ctrl, targ)

class FsimGate(ParaTwoQubitGate):
    def __init__(self, pos, paras):
        super().__init__("fSim", pos, paras)
        self.__theta = paras[0]
        self.__phi   = paras[1]

    def to_nodes(self):
        raise ValueError("The IOP backend does not support fSim gate currently, please use the BAQIS backend.")

    def to_IOP(self):
        raise ValueError("The IOP backend does not support fSim gate currently, please use the BAQIS backend.")