from ..quantum_element import ControlledGate
from abc import ABC
from .matrices import XMatrix, YMatrix, ZMatrix, SMatrix, TMatrix, pmatrix


class _C11Gate(ControlledGate, ABC):
    ct_dims = (1, 1, 2)


class CXGate(_C11Gate):
    name = "CX"

    def __init__(self, ctrl: int, targ: int):
        super().__init__("X", [ctrl], [targ], None, tar_matrix=XMatrix)
        self.symbol = "+"


class CYGate(_C11Gate):
    name = "CY"

    def __init__(self, ctrl: int, targ: int):
        super().__init__("Y", [ctrl], [targ], None, tar_matrix=YMatrix)


class CZGate(_C11Gate):
    name = "CZ"

    def __init__(self, ctrl: int, targ: int):
        super().__init__("Z", [ctrl], [targ], None, tar_matrix=ZMatrix)


class CSGate(_C11Gate):
    name = "CS"

    def __init__(self, ctrl: int, targ: int):
        super().__init__("S", [ctrl], [targ], None, tar_matrix=SMatrix)

    def to_qasm(self):
        return "cp(pi/2) " + "q[%d],q[%d]" % (self.pos[0], self.pos[1])


class CTGate(_C11Gate):
    name = "CT"

    def __init__(self, ctrl: int, targ: int):
        super().__init__("T", [ctrl], [targ], None, tar_matrix=TMatrix)

    def to_qasm(self):
        return "cp(pi/4) " + "q[%d],q[%d]" % (self.pos[0], self.pos[1])


class CPGate(_C11Gate):
    name = "CP"

    def __init__(self, ctrl: int, targ: int, paras):
        super().__init__("P", [ctrl], [targ], paras, tar_matrix=pmatrix(paras))


ControlledGate.register_gate(CXGate)
ControlledGate.register_gate(CYGate)
ControlledGate.register_gate(CZGate)
ControlledGate.register_gate(CSGate)
ControlledGate.register_gate(CTGate)
ControlledGate.register_gate(CPGate)
