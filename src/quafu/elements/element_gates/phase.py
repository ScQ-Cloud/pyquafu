from ..quantum_element import ParaSingleQubitGate
from ._matrices import pmatrix


class PhaseGate(ParaSingleQubitGate):
    def __init__(self, pos: int, paras):
        super().__init__("P", pos, paras, matrix=pmatrix)
