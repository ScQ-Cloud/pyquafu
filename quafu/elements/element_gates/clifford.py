import numpy as np
from quafu.elements.matrices import HMatrix, SMatrix

from ..quantum_gate import FixedGate, QuantumGate, SingleQubitGate

__all__ = ["HGate", "SGate", "SdgGate", "TGate", "TdgGate"]


@QuantumGate.register("h")
class HGate(SingleQubitGate, FixedGate):
    name = "H"
    matrix = HMatrix

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)


@QuantumGate.register("s")
class SGate(SingleQubitGate, FixedGate):
    name = "S"
    matrix = SMatrix

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)


@QuantumGate.register("sdg")
class SdgGate(SingleQubitGate, FixedGate):
    name = "Sdg"
    matrix = SMatrix.conj().T

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)


@QuantumGate.register("t")
class TGate(SingleQubitGate, FixedGate):
    name = "T"
    matrix = np.array([[1.0, 0.0], [0.0, np.exp(1.0j * np.pi / 4)]], dtype=complex)

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)


@QuantumGate.register("tdg")
class TdgGate(SingleQubitGate, FixedGate):
    name = "Tdg"
    matrix = TGate.matrix.conj().T

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)
