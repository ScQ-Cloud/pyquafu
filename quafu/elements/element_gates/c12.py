from ..quantum_gate import ControlledGate, FixedGate
from quafu.elements.matrices import SwapMatrix


class FredkinGate(ControlledGate, FixedGate):
    name = "CSWAP"

    def __init__(self, ctrl: int, targ1: int, targ2: int):
        ControlledGate.__init__(self, "SWAP", [ctrl], [targ1, targ2], None, tar_matrix=SwapMatrix)


ControlledGate.register_gate(FredkinGate)
