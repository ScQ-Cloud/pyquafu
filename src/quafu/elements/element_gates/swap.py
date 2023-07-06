from ..quantum_element import FixedMultiQubitGate
from ._matrices import ISwapMatrix, SwapMatrix


class iSwapGate(FixedMultiQubitGate):
    def __init__(self, q1: int, q2: int):
        super().__init__("iSWAP", [q1, q2], matrix=ISwapMatrix)

    def get_targ_matrix(self, reverse_order=False):
        return self.matrix


class SwapGate(FixedMultiQubitGate):
    def __init__(self, q1: int, q2: int):
        super().__init__("SWAP", [q1, q2], matrix=SwapMatrix)
        self.symbol = "x"

    def get_targ_matrix(self, reverse_order=False):
        return self.matrix
