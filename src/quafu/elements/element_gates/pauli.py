from ..quantum_element import FixedSingleQubitGate
import numpy as np


class IdGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("Id", pos, matrix=np.array([[1., 0.],
                                                     [0., 1.]], dtype=complex))


class HGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("H", pos, matrix=1 / np.sqrt(2) * np.array([[1., 1.],
                                                                     [1., -1.]], dtype=complex))


class XGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("X", pos, matrix=np.array([[0., 1.],
                                                    [1., 0.]], dtype=complex))


class YGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("Y", pos, matrix=np.array([[0., -1.j],
                                                    [1.j, 0.]], dtype=complex))


class ZGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("Z", pos, matrix=np.array([[1., 0.],
                                                    [0., -1.]], dtype=complex))


class WGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("W", pos, matrix=np.zeros((2, 2), dtype=complex))
        self.matrix = (XGate(0).matrix + YGate(0).matrix) / np.sqrt(2)

    def to_qasm(self):
        return "rz(-pi/4) q[%d];\nrx(pi) q[%d];\nrz(pi/4) q[%d]" % (self.pos, self.pos, self.pos)


class SWGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("SW", pos, matrix=np.zeros((2, 2), dtype=complex))
        self.matrix = np.array([[0.5 + 0.5j, -np.sqrt(0.5) * 1j],
                                [np.sqrt(0.5), 0.5 + 0.5j]], dtype=complex)
        self.symbol = "√W"

    def to_qasm(self):
        return "rz(-pi/4) q[%d];\nrx(pi/2) q[%d];\nrz(pi/4) q[%d]" % (self.pos, self.pos, self.pos)