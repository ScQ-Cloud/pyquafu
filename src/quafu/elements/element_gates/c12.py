from ..quantum_element import ControlledGate
from .matrices import SwapMatrix


class FredkinGate(ControlledGate):
    name = "CSWAP"

    def __init__(self, ctrl: int, targ1: int, targ2: int):
        super().__init__("SWAP", [ctrl], [targ1, targ2], None, tar_matrix=SwapMatrix)


ControlledGate.register_gate(FredkinGate)
