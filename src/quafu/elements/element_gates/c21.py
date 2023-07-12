from ..quantum_element import ControlledGate
from .matrices import XMatrix


class ToffoliGate(ControlledGate):
    name = "CCX"

    def __init__(self, ctrl1: int, ctrl2: int, targ: int):
        super().__init__("X", [ctrl1, ctrl2], [targ], None, tar_matrix=XMatrix)


ControlledGate.register_gate(ToffoliGate)

