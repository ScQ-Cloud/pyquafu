import numpy as np

from quafu.elements.matrices import XMatrix, YMatrix, ZMatrix, WMatrix, SWMatrix
from quafu.elements.quantum_gate import FixedGate, SingleQubitGate

__all__ = ['IdGate', 'XGate', 'YGate', 'ZGate',
           'WGate', 'SWGate',
           'SXGate', 'SYGate', 'SXdgGate', 'SYdgGate']


class IdGate(FixedGate, SingleQubitGate):
    name = "Id"
    matrix = XMatrix

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)


class XGate(FixedGate, SingleQubitGate):
    name = "X"
    matrix = XMatrix

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)


class YGate(FixedGate, SingleQubitGate):
    name = "Y"
    matrix = YMatrix

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)


class ZGate(FixedGate, SingleQubitGate):
    name = "Z"
    matrix = ZMatrix

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)


class WGate(FixedGate, SingleQubitGate):
    """ (X+Y)/sqrt(2) """
    name = "W"
    matrix = WMatrix

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)
        self.symbol = "W"

    def to_qasm(self):
        return "rz(-pi/4) q[%d];\nrx(pi) q[%d];\nrz(pi/4) q[%d]" % (self.pos, self.pos, self.pos)


class SWGate(FixedGate, SingleQubitGate):
    name = "SW"
    matrix = SWMatrix

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)
        self.symbol = "√W"

    def to_qasm(self):
        return "rz(-pi/4) q[%d];\nrx(pi/2) q[%d];\nrz(pi/4) q[%d]" % (self.pos, self.pos, self.pos)


class SXGate(FixedGate, SingleQubitGate):
    name = "SX"
    matrix = np.array([[0.5 + 0.5j, 0.5 - 0.5j],
                       [0.5 - 0.5j, 0.5 + 0.5j]], dtype=complex)

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)


class SXdgGate(FixedGate, SingleQubitGate):
    name = "SXdg"
    matrix = SXGate.matrix.conj().T

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)
        self.symbol = "√X"


class SYGate(FixedGate, SingleQubitGate):
    name = "SY"
    matrix = np.array([[0.5 + 0.5j, -0.5 - 0.5j],
                       [0.5 + 0.5j, 0.5 + 0.5j]], dtype=complex)

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)
        self.symbol = "√Y"

    def to_qasm(self):
        return "ry(pi/2) q[%d];" % self.pos


class SYdgGate(FixedGate, SingleQubitGate):
    name = "SYdg"
    matrix = SYGate.matrix.conj().T

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)
        self.symbol = "√Y†"

    def to_qasm(self):
        return "ry(-pi/2) q[%d]" % self.pos


SingleQubitGate.register_gate(IdGate)
SingleQubitGate.register_gate(XGate)
SingleQubitGate.register_gate(YGate)
SingleQubitGate.register_gate(ZGate)
SingleQubitGate.register_gate(WGate)
SingleQubitGate.register_gate(SWGate)
SingleQubitGate.register_gate(SXGate)
SingleQubitGate.register_gate(SXdgGate)
SingleQubitGate.register_gate(SYGate)
SingleQubitGate.register_gate(SYdgGate)
