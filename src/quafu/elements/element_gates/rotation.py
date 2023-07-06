from ..quantum_element import ParaMultiQubitGate, ParaSingleQubitGate
from ._matrices import rxmatrix, rymatrix, rzmatrix, rxxmatrix, ryymatrix, rzzmatrix


class RXGate(ParaSingleQubitGate):
    def __init__(self, pos: int, paras):
        super().__init__("RX", pos, paras, matrix=rxmatrix)


class RYGate(ParaSingleQubitGate):
    def __init__(self, pos: int, paras):
        super().__init__("RY", pos, paras, matrix=rymatrix)


class RZGate(ParaSingleQubitGate):
    def __init__(self, pos: int, paras):
        super().__init__("RZ", pos, paras, matrix=rzmatrix)


class RXXGate(ParaMultiQubitGate):
    def __init__(self, q1: int, q2: int, theta):
        super().__init__("RXX", [q1, q2], theta, matrix=rxxmatrix(theta))

    def get_targ_matrix(self, reverse_order=False):
        return self.matrix


class RYYGate(ParaMultiQubitGate):
    def __init__(self, q1: int, q2: int, theta):
        super().__init__("RYY", [q1, q2], theta, matrix=ryymatrix(theta))

    def get_targ_matrix(self, reverse_order=False):
        return self.matrix


class RZZGate(ParaMultiQubitGate):
    def __init__(self, q1: int, q2: int, theta):
        super().__init__("RZZ", [q1, q2], theta, matrix=rzzmatrix(theta))

    def get_targ_matrix(self, reverse_order=False):
        return self.matrix
