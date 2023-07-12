from ..quantum_element import FixedMultiQubitGate
from .matrices import ISwapMatrix, SwapMatrix


class ISwapGate(FixedMultiQubitGate):
    name = "iSWAP"
    matrix = ISwapMatrix

    def __init__(self, q1: int, q2: int):
        super().__init__([q1, q2])

    def get_targ_matrix(self, reverse_order=False):
        return self.matrix


class SwapGate(FixedMultiQubitGate):
    name = "SWAP"
    matrix = SwapMatrix

    def __init__(self, q1: int, q2: int):
        super().__init__([q1, q2])
        self.symbol = "x"

    def get_targ_matrix(self, reverse_order=False):
        return self.matrix


FixedMultiQubitGate.register_gate(ISwapGate)
FixedMultiQubitGate.register_gate(SwapGate)
