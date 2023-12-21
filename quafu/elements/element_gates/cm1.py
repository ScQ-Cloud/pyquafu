from quafu.elements.matrices import XMatrix, YMatrix, ZMatrix

from ..quantum_gate import ControlledGate, FixedGate, QuantumGate

__all__ = ["MCXGate", "MCYGate", "MCZGate", "ToffoliGate"]


@QuantumGate.register("mcx")
class MCXGate(ControlledGate, FixedGate):
    name = "MCX"

    def __init__(self, ctrls, targ: int):
        ControlledGate.__init__(self, "X", ctrls, [targ], None, tar_matrix=XMatrix)


@QuantumGate.register("mcy")
class MCYGate(ControlledGate, FixedGate):
    name = "MCY"

    def __init__(self, ctrls, targ: int):
        ControlledGate.__init__(self, "Y", ctrls, [targ], None, tar_matrix=YMatrix)


@QuantumGate.register("mcz")
class MCZGate(ControlledGate, FixedGate):
    name = "MCZ"

    def __init__(self, ctrls, targ: int):
        ControlledGate.__init__(self, "Z", ctrls, [targ], None, tar_matrix=ZMatrix)


@QuantumGate.register("ccx")
class ToffoliGate(ControlledGate, FixedGate):
    name = "CCX"

    def __init__(self, ctrl1: int, ctrl2: int, targ: int):
        ControlledGate.__init__(
            self, "X", [ctrl1, ctrl2], [targ], None, tar_matrix=XMatrix
        )
