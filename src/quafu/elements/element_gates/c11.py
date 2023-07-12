from ..quantum_element import ControlledGate
from .matrices import XMatrix, YMatrix, ZMatrix, SMatrix, TMatrix, pmatrix


class CXGate(ControlledGate):
    name = "CX"

    def __init__(self, ctrl: int, targ: int):
        super().__init__("X", [ctrl], [targ], None, tar_matrix=XMatrix)
        self.symbol = "+"


class CYGate(ControlledGate):
    name = "CY"

    def __init__(self, ctrl: int, targ: int):
        super().__init__("Y", [ctrl], [targ], None, tar_matrix=YMatrix)


class CZGate(ControlledGate):
    name = "CZ"

    def __init__(self, ctrl: int, targ: int):
        super().__init__("Z", [ctrl], [targ], None, tar_matrix=ZMatrix)


class CSGate(ControlledGate):
    name = "CS"

    def __init__(self, ctrl: int, targ: int):
        super().__init__("S", [ctrl], [targ], None, tar_matrix=SMatrix)

    def to_qasm(self):
        return "cp(pi/2) " + "q[%d],q[%d]" % (self.pos[0], self.pos[1])


class CTGate(ControlledGate):
    name = "CT"

    def __init__(self, ctrl: int, targ: int):
        super().__init__("T", [ctrl], [targ], None, tar_matrix=TMatrix)

    def to_qasm(self):
        return "cp(pi/4) " + "q[%d],q[%d]" % (self.pos[0], self.pos[1])


class CPGate(ControlledGate):
    name = "CP"

    def __init__(self, ctrl: int, targ: int, paras):
        super().__init__("P", [ctrl], [targ], paras, tar_matrix=pmatrix(paras))
