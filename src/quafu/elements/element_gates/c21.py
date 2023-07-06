from ..quantum_element import ControlledGate
from ._matrices import XMatrix


class ToffoliGate(ControlledGate):
    def __init__(self, ctrl1: int, ctrl2: int, targ: int):
        super().__init__("CCX", "X", [ctrl1, ctrl2], [targ], None, matrix=XMatrix)
