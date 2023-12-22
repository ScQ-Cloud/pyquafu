from typing import Dict

from quafu.elements.matrices import ISwapMatrix, SwapMatrix

from ..quantum_gate import FixedGate, MultiQubitGate, QuantumGate

__all__ = ["ISwapGate", "SwapGate"]


@QuantumGate.register("iswap")
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
        return {"pos": self.pos}


@QuantumGate.register("swap")
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
        return {"pos": self.pos}
