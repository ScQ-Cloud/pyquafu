from ..quantum_element import ParaMultiQubitGate, ParaSingleQubitGate
from .matrices import rx_mat, ry_mat, rz_mat, rxx_mat, ryy_mat, rzz_mat


class RXGate(ParaSingleQubitGate):
    name = "RX"

    def __init__(self, pos: int, paras: float = 0.):
        super().__init__(pos, paras)

    @property
    def matrix(self):
        return rx_mat(self.paras)


class RYGate(ParaSingleQubitGate):
    name = "RY"

    def __init__(self, pos: int, paras: float = 0.):
        super().__init__(pos, paras)

    @property
    def matrix(self):
        return ry_mat(self.paras)


class RZGate(ParaSingleQubitGate):
    name = "RZ"

    def __init__(self, pos: int, paras: float = 0.):
        super().__init__(pos, paras)

    @property
    def matrix(self):
        return rz_mat(self.paras)


class RXXGate(ParaMultiQubitGate):
    name = "RXX"

    def __init__(self, q1: int, q2: int, theta: float = 0.):
        super().__init__([q1, q2], paras=theta)

    @property
    def matrix(self):
        return rxx_mat(self.paras)

    def get_targ_matrix(self, reverse_order=False):
        return self.matrix


class RYYGate(ParaMultiQubitGate):
    name = "RYY"

    def __init__(self, q1: int, q2: int, theta: float = 0.):
        super().__init__([q1, q2], paras=theta)

    @property
    def matrix(self):
        return ryy_mat(self.paras)

    def get_targ_matrix(self, reverse_order=False):
        return self.matrix


class RZZGate(ParaMultiQubitGate):
    name = "RZZ"

    def __init__(self, q1: int, q2: int, theta: float = 0.):
        super().__init__([q1, q2], paras=theta)

    @property
    def matrix(self):
        return rzz_mat(self.paras)

    def get_targ_matrix(self, reverse_order=False):
        return self.matrix


ParaSingleQubitGate.register_gate(RXGate)
ParaSingleQubitGate.register_gate(RYGate)
ParaSingleQubitGate.register_gate(RZGate)
ParaMultiQubitGate.register_gate(RXXGate)
ParaMultiQubitGate.register_gate(RYYGate)
ParaMultiQubitGate.register_gate(RZZGate)
