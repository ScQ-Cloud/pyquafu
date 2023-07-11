from ..quantum_element import ControlledGate
from ._matrices import XMatrix, YMatrix, ZMatrix, SMatrix, TMatrix, pmatrix


class CXGate(ControlledGate):
    def __init__(self, ctrl: int, targ: int):
        super().__init__("CX", "X", [ctrl], [targ], None, matrix=XMatrix)
        self.symbol = "+"


class CYGate(ControlledGate):
    def __init__(self, ctrl: int, targ: int):
        super().__init__("CY", "Y", [ctrl], [targ], None, matrix=YMatrix)


class CZGate(ControlledGate):
    def __init__(self, ctrl: int, targ: int):
        super().__init__("CZ", "Z", [ctrl], [targ], None, matrix=ZMatrix)


class CSGate(ControlledGate):
    def __init__(self, ctrl: int, targ: int):
        super().__init__("CS", "S", [ctrl], [targ], None, matrix=SMatrix)

    def to_qasm(self):
        return "cp(pi/2) " + "q[%d],q[%d]" % (self.pos[0], self.pos[1])


class CTGate(ControlledGate):
    def __init__(self, ctrl: int, targ: int):
        super().__init__("CT", "T", [ctrl], [targ], None, matrix=TMatrix)

    def to_qasm(self):
        return "cp(pi/4) " + "q[%d],q[%d]" % (self.pos[0], self.pos[1])


class CPGate(ControlledGate):
    def __init__(self, ctrl: int, targ: int, paras):
        super().__init__("CP", "P", [ctrl], [targ], paras, matrix=pmatrix(paras))
