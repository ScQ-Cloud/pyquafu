from ..quantum_gate import SingleQubitGate
from quafu.elements.matrices import pmatrix
from typing import Dict


class PhaseGate(SingleQubitGate):
    name = "P"

    def __init__(self, pos: int, paras: float = 0.):
        super().__init__(pos, paras=paras)

    @property
    def matrix(self):
        return pmatrix(self.paras)

    @property
    def named_paras(self) -> Dict:
        return {'phase': self.paras}


SingleQubitGate.register_gate(PhaseGate)
