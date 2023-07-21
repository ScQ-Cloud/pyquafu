from typing import List, Union

from .matrices import XMatrix, YMatrix, ZMatrix
from ..quantum_element import ControlledGate, SingleQubitGate, MultiQubitGate


class MCXGate(ControlledGate):
    name = "MCX"

    def __init__(self, ctrls, targ: int):
        super().__init__("X", ctrls, [targ], None, tar_matrix=XMatrix)


class MCYGate(ControlledGate):
    name = "MCY"

    def __init__(self, ctrls, targ: int):
        super().__init__("Y", ctrls, [targ], None, tar_matrix=YMatrix)


class MCZGate(ControlledGate):
    name = "MCZ"

    def __init__(self, ctrls, targ: int):
        super().__init__("Z", ctrls, [targ], None, tar_matrix=ZMatrix)


class ControlledU(ControlledGate):
    """ Controlled gate class, where the matrix act non-trivially on target qubits"""
    name = 'CU'

    def __init__(self, ctrls: List[int], u: Union[SingleQubitGate, MultiQubitGate]):
        self.targ_gate = u
        targs = u.pos
        if isinstance(targs, int):
            targs = [targs]

        super().__init__(u.name, ctrls, targs, u.paras, tar_matrix=self.targ_gate.get_targ_matrix())

    def get_targ_matrix(self, reverse_order=False):
        return self.targ_gate.get_targ_matrix(reverse_order)


ControlledGate.register_gate(MCXGate)
ControlledGate.register_gate(MCYGate)
ControlledGate.register_gate(MCZGate)
ControlledGate.register_gate(ControlledU)
