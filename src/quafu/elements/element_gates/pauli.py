from .matrices import XMatrix, YMatrix, ZMatrix, HMatrix, WMatrix, SWMatrix
from ..quantum_element import FixedSingleQubitGate


class IdGate(FixedSingleQubitGate):
    name = "Id"
    matrix = XMatrix

    def __init__(self, pos: int):
        super().__init__(pos)


class HGate(FixedSingleQubitGate):
    name = "H"
    matrix = HMatrix

    def __init__(self, pos: int):
        super().__init__(pos)


class XGate(FixedSingleQubitGate):
    name = "X"
    matrix = XMatrix

    def __init__(self, pos: int):
        super().__init__(pos)


class YGate(FixedSingleQubitGate):
    name = "Y"
    matrix = YMatrix

    def __init__(self, pos: int):
        super().__init__(pos)


class ZGate(FixedSingleQubitGate):
    name = "Z"
    matrix = ZMatrix

    def __init__(self, pos: int):
        super().__init__(pos)


class WGate(FixedSingleQubitGate):
    name = "W"
    matrix = WMatrix

    def __init__(self, pos: int):
        super().__init__(pos)
        self.symbol = "W"

    def to_qasm(self):
        return "rz(-pi/4) q[%d];\nrx(pi) q[%d];\nrz(pi/4) q[%d]" % (self.pos, self.pos, self.pos)


class SWGate(FixedSingleQubitGate):
    name = "SW"
    matrix = SWMatrix

    def __init__(self, pos: int):
        super().__init__(pos)
        self.symbol = "√W"

    def to_qasm(self):
        return "rz(-pi/4) q[%d];\nrx(pi/2) q[%d];\nrz(pi/4) q[%d]" % (self.pos, self.pos, self.pos)
