from ..quantum_element import FixedSingleQubitGate
import numpy as np


class SGate(FixedSingleQubitGate):
    name = "S"
    matrix = np.array([[1., 0.],
                       [0., 1.j]], dtype=complex)

    def __init__(self, pos: int):
        super().__init__(pos)


class SdgGate(FixedSingleQubitGate):
    name = "Sdg"
    matrix = SGate.matrix.conj().T

    def __init__(self, pos: int):
        super().__init__(pos)


class TGate(FixedSingleQubitGate):
    name = "T"
    matrix = np.array([[1., 0.],
                       [0., np.exp(1.j * np.pi / 4)]], dtype=complex)

    def __init__(self, pos: int):
        super().__init__(pos)


class TdgGate(FixedSingleQubitGate):
    name = "Tdg"
    matrix = TGate.matrix.conj().T

    def __init__(self, pos: int):
        super().__init__(pos)


class SXGate(FixedSingleQubitGate):
    name = "SX"
    matrix = np.array([[0.5 + 0.5j, 0.5 - 0.5j],
                       [0.5 - 0.5j, 0.5 + 0.5j]], dtype=complex)

    def __init__(self, pos: int):
        super().__init__(pos)


class SXdgGate(FixedSingleQubitGate):
    name = "SXdg"
    matrix = SXGate.matrix.conj().T

    def __init__(self, pos: int):
        super().__init__(pos)
        self.symbol = "√X"


class SYGate(FixedSingleQubitGate):
    name = "SY"
    matrix = np.array([[0.5 + 0.5j, -0.5 - 0.5j],
                       [0.5 + 0.5j, 0.5 + 0.5j]], dtype=complex)

    def __init__(self, pos: int):
        super().__init__(pos)
        self.symbol = "√Y"


class SYdgGate(FixedSingleQubitGate):
    name = "SYdg"
    matrix = SYGate.matrix.conj().T

    def __init__(self, pos: int):
        super().__init__(pos)
        self.symbol = "√Y†"

    def to_qasm(self):
        # TODO: this seems incorrect
        return "ry(pi/2) q[%d]" % self.pos
