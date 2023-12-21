from abc import ABC
from typing import Dict

from quafu.elements.matrices import SMatrix, TMatrix, XMatrix, YMatrix, ZMatrix, pmatrix

from ..quantum_gate import ControlledGate, FixedGate, QuantumGate

__all__ = ["CXGate", "CYGate", "CZGate", "CSGate", "CTGate", "CPGate"]


class _C11Gate(ControlledGate, ABC):
    ct_dims = (1, 1, 2)


@QuantumGate.register("cx")
class CXGate(_C11Gate, FixedGate):
    name = "CX"

    def __init__(self, ctrl: int, targ: int):
        _C11Gate.__init__(self, "X", [ctrl], [targ], None, tar_matrix=XMatrix)
        self.symbol = "+"


@QuantumGate.register("cy")
class CYGate(_C11Gate, FixedGate):
    name = "CY"

    def __init__(self, ctrl: int, targ: int):
        _C11Gate.__init__(self, "Y", [ctrl], [targ], None, tar_matrix=YMatrix)


@QuantumGate.register("cz")
class CZGate(_C11Gate, FixedGate):
    name = "CZ"

    def __init__(self, ctrl: int, targ: int):
        _C11Gate.__init__(self, "Z", [ctrl], [targ], None, tar_matrix=ZMatrix)


@QuantumGate.register("cs")
class CSGate(_C11Gate, FixedGate):
    name = "CS"

    def __init__(self, ctrl: int, targ: int):
        _C11Gate.__init__(self, "S", [ctrl], [targ], None, tar_matrix=SMatrix)

    def to_qasm(self):
        return "cp(pi/2) " + "q[%d],q[%d]" % (self.pos[0], self.pos[1])


@QuantumGate.register("ct")
class CTGate(_C11Gate, FixedGate):
    name = "CT"

    def __init__(self, ctrl: int, targ: int):
        _C11Gate.__init__(self, "T", [ctrl], [targ], None, tar_matrix=TMatrix)

    def to_qasm(self):
        return "cp(pi/4) " + "q[%d],q[%d]" % (self.pos[0], self.pos[1])


@QuantumGate.register("cp")
class CPGate(_C11Gate):
    name = "CP"

    def __init__(self, ctrl: int, targ: int, paras):
        _C11Gate.__init__(self, "P", [ctrl], [targ], paras, tar_matrix=pmatrix(paras))

    @property
    def named_paras(self) -> Dict:
        return {"theta": self.paras}
