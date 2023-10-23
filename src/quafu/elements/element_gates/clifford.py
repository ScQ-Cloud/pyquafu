import numpy as np

from quafu.elements.matrices import HMatrix, SMatrix
from ..quantum_gate import SingleQubitGate, FixedGate

__all__ = ['HGate', 'SGate', 'SdgGate', 'TGate', 'TdgGate']


class HGate(SingleQubitGate, FixedGate):
    name = "H"
    matrix = HMatrix

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)


class SGate(SingleQubitGate, FixedGate):
    name = "S"
    matrix = SMatrix

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)


class SdgGate(SingleQubitGate, FixedGate):
    name = "Sdg"
    matrix = SMatrix.conj().T

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)


class TGate(SingleQubitGate, FixedGate):
    name = "T"
    matrix = np.array([[1., 0.],
                       [0., np.exp(1.j * np.pi / 4)]], dtype=complex)

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)


class TdgGate(SingleQubitGate, FixedGate):
    name = "Tdg"
    matrix = TGate.matrix.conj().T

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)


FixedGate.register_gate(HGate)
FixedGate.register_gate(SGate)
FixedGate.register_gate(SdgGate)
FixedGate.register_gate(TGate)
FixedGate.register_gate(TdgGate)
