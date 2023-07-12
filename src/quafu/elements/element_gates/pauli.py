import numpy as np

from .matrices import XMatrix, YMatrix, ZMatrix, WMatrix, SWMatrix
from ..quantum_element import FixedSingleQubitGate


class IdGate(FixedSingleQubitGate):
    name = "Id"
    matrix = XMatrix

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


class SXGate(FixedSingleQubitGate):
    name = "SX"
    matrix = np.array([[0.5 + 0.5j, 0.5 - 0.5j],
                       [0.5 - 0.5j, 0.5 + 0.5j]], dtype=complex)

    def __init__(self, pos: int):
        super().__init__(pos)


class SXdgGate(FixedSingleQubitGate):
    name = "SXdg"
    matrix = SXGate.matrix.conj().T

    def __init__(self, pos: int):
        super().__init__(pos)
        self.symbol = "√X"


class SYGate(FixedSingleQubitGate):
    name = "SY"
    matrix = np.array([[0.5 + 0.5j, -0.5 - 0.5j],
                       [0.5 + 0.5j, 0.5 + 0.5j]], dtype=complex)

    def __init__(self, pos: int):
        super().__init__(pos)
        self.symbol = "√Y"


class SYdgGate(FixedSingleQubitGate):
    name = "SYdg"
    matrix = SYGate.matrix.conj().T

    def __init__(self, pos: int):
        super().__init__(pos)
        self.symbol = "√Y†"

    def to_qasm(self):
        # TODO: this seems incorrect
        return "ry(pi/2) q[%d]" % self.pos


FixedSingleQubitGate.register_gate(IdGate)
FixedSingleQubitGate.register_gate(XGate)
FixedSingleQubitGate.register_gate(YGate)
FixedSingleQubitGate.register_gate(ZGate)
FixedSingleQubitGate.register_gate(WGate)
FixedSingleQubitGate.register_gate(SWGate)
FixedSingleQubitGate.register_gate(SXGate)
FixedSingleQubitGate.register_gate(SXdgGate)
FixedSingleQubitGate.register_gate(SYGate)
FixedSingleQubitGate.register_gate(SYdgGate)
