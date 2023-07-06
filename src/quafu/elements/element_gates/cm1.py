from ._matrices import XMatrix, YMatrix, ZMatrix
from ..quantum_element import ControlledGate


class MCXGate(ControlledGate):
    def __init__(self, ctrls, targ: int):
        super().__init__("MCX", "X", ctrls, [targ], None, matrix=XMatrix)


class MCYGate(ControlledGate):
    def __init__(self, ctrls, targ: int):
        super().__init__("MCY", "Y", ctrls, [targ], None, matrix=YMatrix)


class MCZGate(ControlledGate):
    def __init__(self, ctrls, targ: int):
        super().__init__("MCZ", "Z", ctrls, [targ], None, matrix=ZMatrix)
