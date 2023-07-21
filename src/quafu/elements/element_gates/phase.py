from ..quantum_element import ParaSingleQubitGate
from .matrices import pmatrix


class PhaseGate(ParaSingleQubitGate):
    name = "P"

    def __init__(self, pos: int, paras: float = 0.):
        super().__init__(pos, paras=paras)

    @property
    def matrix(self):
        return pmatrix(self.paras)


ParaSingleQubitGate.register_gate(PhaseGate)
