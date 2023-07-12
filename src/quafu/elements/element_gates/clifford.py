import numpy as np

from quafu.elements.element_gates.matrices import HMatrix
from quafu.elements.quantum_element import FixedSingleQubitGate


class HGate(FixedSingleQubitGate):
    name = "H"
    matrix = HMatrix

    def __init__(self, pos: int):
        super().__init__(pos)


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


FixedSingleQubitGate.register_gate(HGate)
FixedSingleQubitGate.register_gate(SGate)
FixedSingleQubitGate.register_gate(SdgGate)
FixedSingleQubitGate.register_gate(TGate)
FixedSingleQubitGate.register_gate(TdgGate)
