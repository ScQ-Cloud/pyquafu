from ..quantum_gate import FixedGate, MultiQubitGate
from quafu.elements.matrices import ISwapMatrix, SwapMatrix

from typing import Dict

__all__ = ['ISwapGate', 'SwapGate']


class ISwapGate(FixedGate, MultiQubitGate):
    name = "iSWAP"
    matrix = ISwapMatrix

    def __init__(self, q1: int, q2: int):
        super().__init__([q1, q2])
        self.symbol = "(x)"

    def get_targ_matrix(self, reverse_order=False):
        return self.matrix

    @property
    def named_pos(self) -> Dict:
        return {'pos': self.pos}


class SwapGate(FixedGate, MultiQubitGate):
    name = "SWAP"
    matrix = SwapMatrix

    def __init__(self, q1: int, q2: int):
        super().__init__([q1, q2])
        self.symbol = "x"

    def get_targ_matrix(self, reverse_order=False):
        return self.matrix

    @property
    def named_pos(self) -> Dict:
        return {'pos': self.pos}


FixedGate.register_gate(ISwapGate)
FixedGate.register_gate(SwapGate)
