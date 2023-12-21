from typing import Dict

from quafu.elements.matrices import (
    pmatrix,
    rx_mat,
    rxx_mat,
    ry_mat,
    ryy_mat,
    rz_mat,
    rzz_mat,
)

from ..quantum_gate import ParametricGate, QuantumGate, SingleQubitGate

__all__ = ["RXGate", "RYGate", "RZGate", "RXXGate", "RYYGate", "RZZGate", "PhaseGate"]


@QuantumGate.register("rx")
class RXGate(ParametricGate, SingleQubitGate):
    name = "RX"

    def __init__(self, pos: int, paras: float = 0.0):
        ParametricGate.__init__(self, pos, paras=paras)

    @property
    def matrix(self):
        return rx_mat(self.paras)


@QuantumGate.register("ry")
class RYGate(ParametricGate, SingleQubitGate):
    name = "RY"

    def __init__(self, pos: int, paras: float = 0.0):
        ParametricGate.__init__(self, pos, paras=paras)

    @property
    def matrix(self):
        return ry_mat(self.paras)


@QuantumGate.register("rz")
class RZGate(ParametricGate, SingleQubitGate):
    name = "RZ"

    def __init__(self, pos: int, paras: float = 0.0):
        ParametricGate.__init__(self, pos, paras=paras)

    @property
    def matrix(self):
        return rz_mat(self.paras)


@QuantumGate.register("rxx")
class RXXGate(ParametricGate):
    name = "RXX"

    def __init__(self, q1: int, q2: int, paras: float = 0.0):
        ParametricGate.__init__(self, [q1, q2], paras=paras)

    @property
    def matrix(self):
        return rxx_mat(self.paras)

    @property
    def named_pos(self) -> Dict:
        return {"pos": self.pos}


@QuantumGate.register("ryy")
class RYYGate(ParametricGate):
    name = "RYY"

    def __init__(self, q1: int, q2: int, paras: float = 0.0):
        ParametricGate.__init__(self, [q1, q2], paras=paras)

    @property
    def matrix(self):
        return ryy_mat(self.paras)

    @property
    def named_pos(self) -> Dict:
        return {"pos": self.pos}


@QuantumGate.register("rzz")
class RZZGate(ParametricGate):
    name = "RZZ"

    def __init__(self, q1: int, q2: int, paras: float = 0.0):
        ParametricGate.__init__(self, [q1, q2], paras=paras)

    @property
    def matrix(self):
        return rzz_mat(self.paras)

    @property
    def named_pos(self) -> Dict:
        return {"pos": self.pos}


@SingleQubitGate.register(name="p")
class PhaseGate(SingleQubitGate):
    """Ally of rz gate, but with a different name and global phase."""

    name = "P"

    def __init__(self, pos: int, paras: float = 0.0):
        super().__init__(pos, paras=paras)

    @property
    def matrix(self):
        return pmatrix(self.paras)

    @property
    def named_paras(self) -> Dict:
        return {"phase": self.paras}
