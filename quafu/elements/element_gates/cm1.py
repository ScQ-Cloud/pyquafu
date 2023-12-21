from quafu.elements.matrices import XMatrix, YMatrix, ZMatrix

from ..quantum_gate import ControlledGate, FixedGate

__all__ = ["MCXGate", "MCYGate", "MCZGate", "ToffoliGate"]


class MCXGate(ControlledGate, FixedGate):
    name = "MCX"

    def __init__(self, ctrls, targ: int):
        ControlledGate.__init__(self, "X", ctrls, [targ], None, tar_matrix=XMatrix)


class MCYGate(ControlledGate, FixedGate):
    name = "MCY"

    def __init__(self, ctrls, targ: int):
        ControlledGate.__init__(self, "Y", ctrls, [targ], None, tar_matrix=YMatrix)


class MCZGate(ControlledGate, FixedGate):
    name = "MCZ"

    def __init__(self, ctrls, targ: int):
        ControlledGate.__init__(self, "Z", ctrls, [targ], None, tar_matrix=ZMatrix)


class ToffoliGate(ControlledGate, FixedGate):
    name = "CCX"

    def __init__(self, ctrl1: int, ctrl2: int, targ: int):
        ControlledGate.__init__(
            self, "X", [ctrl1, ctrl2], [targ], None, tar_matrix=XMatrix
        )


ControlledGate.register_gate(MCXGate)
ControlledGate.register_gate(MCYGate)
ControlledGate.register_gate(MCZGate)
ControlledGate.register_gate(ToffoliGate)

# deprecated

# class ControlledU(ControlledGate):
#     """ Controlled gate class, where the matrix act non-trivially on target qubits"""
#     name = 'CU'
#
#     def __init__(self, ctrls: List[int], u: Union[SingleQubitGate, MultiQubitGate]):
#         self.targ_gate = u
#         targs = u.pos
#         if isinstance(targs, int):
#             targs = [targs]
#
#         ControlledGate.__init__(self, u.name, ctrls, targs, u.paras, tar_matrix=self.targ_gate.get_targ_matrix())
#
#     def get_targ_matrix(self, reverse_order=False):
#         return self.targ_gate.get_targ_matrix(reverse_order)
# ControlledGate.register_gate(ControlledU)
