from quafu.elements.matrices import SwapMatrix

from ..quantum_gate import ControlledGate, FixedGate, QuantumGate


@QuantumGate.register("cswap")
class FredkinGate(ControlledGate, FixedGate):
    name = "CSWAP"

    def __init__(self, ctrl: int, targ1: int, targ2: int):
        ControlledGate.__init__(
            self, "SWAP", [ctrl], [targ1, targ2], None, tar_matrix=SwapMatrix
        )
