from ..quantum_element import FixedSingleQubitGate
import numpy as np


class SGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("S", pos, matrix=np.array([[1., 0.],
                                                    [0., 1.j]], dtype=complex))


class SdgGate(FixedSingleQubitGate):
    def __init__(sell, pos: int):
        super().__init__("Sdg", pos, matrix=np.array([[1., 0.],
                                                      [0., -1.j]], dtype=complex))


class TGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("T", pos, matrix=np.array([[1., 0.],
                                                    [0., np.exp(1.j * np.pi / 4)]], dtype=complex))


class TdgGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("Tdg", pos, matrix=np.array([[1., 0.],
                                                      [0, np.exp(-1.j * np.pi / 4)]], dtype=complex))


class SXGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("SX", pos, matrix=np.zeros((2, 2), dtype=complex))
        self.matrix = np.array([[0.5 + 0.5j, 0.5 - 0.5j],
                                [0.5 - 0.5j, 0.5 + 0.5j]])
        self.symbol = "√X"


class SXdgGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("SXdg", pos, matrix=np.zeros((2, 2), dtype=complex))
        matrix = np.array([[0.5 - 0.5j, 0.5 + 0.5j],
                           [0.5 + 0.5j, 0.5 - 0.5j]])
        self.matrix = matrix
        self.symbol = "√X†"


class SYGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("SY", pos, matrix=np.zeros((2, 2), dtype=complex))
        self.matrix = np.array([[0.5 + 0.5j, -0.5 - 0.5j],
                                [0.5 + 0.5j, 0.5 + 0.5j]])
        self.symbol = "√Y"

    def to_qasm(self):
        # TODO: this is not correct
        return "ry(pi/2) q[%d]" % self.pos


class SYdgGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("SY", pos, matrix=np.zeros((2, 2), dtype=complex))
        self.matrix = np.array([[0.5 - 0.5j, -0.5 + 0.5j],
                                [0.5 - 0.5j, 0.5 - 0.5j]])
        self.symbol = "√Y†"
