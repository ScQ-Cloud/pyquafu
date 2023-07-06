from ..quantum_element import ControlledGate
from ._matrices import SwapMatrix


class FredkinGate(ControlledGate):
    def __init__(self, ctrl: int, targ1: int, targ2: int):
        super().__init__("CSWAP", "SWAP", [ctrl], [targ1, targ2], None, matrix=SwapMatrix)
