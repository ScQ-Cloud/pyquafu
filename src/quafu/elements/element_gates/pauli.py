from ._matrices import XMatrix, YMatrix, ZMatrix, HMatrix, WMatrix, SWMatrix
from ..quantum_element import FixedSingleQubitGate


class IdGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("Id", pos, matrix=XMatrix)


class HGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("H", pos, matrix=HMatrix)


class XGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("X", pos, matrix=XMatrix)


class YGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("Y", pos, matrix=YMatrix)


class ZGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("Z", pos, matrix=ZMatrix)


class WGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("W", pos, matrix=WMatrix)
        self.symbol = "W"

    def to_qasm(self):
        return "rz(-pi/4) q[%d];\nrx(pi) q[%d];\nrz(pi/4) q[%d]" % (self.pos, self.pos, self.pos)


class SWGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("SW", pos, matrix=SWMatrix)
        self.symbol = "√W"

    def to_qasm(self):
        return "rz(-pi/4) q[%d];\nrx(pi/2) q[%d];\nrz(pi/4) q[%d]" % (self.pos, self.pos, self.pos)
