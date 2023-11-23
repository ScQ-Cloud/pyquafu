from typing import Dict

from quafu.elements.matrices import rx_mat, ry_mat, rz_mat, rxx_mat, ryy_mat, rzz_mat
from ..quantum_gate import QuantumGate, SingleQubitGate, ParametricGate

__all__ = ['RXGate', 'RYGate', 'RZGate', 'RXXGate', 'RYYGate', 'RZZGate']


class RXGate(ParametricGate, SingleQubitGate):
    name = "RX"
    
    def __init__(self, pos: int, paras: float = 0.):
        ParametricGate.__init__(self, pos, paras=paras)

    @property
    def matrix(self):
        return rx_mat(self.paras)


class RYGate(ParametricGate, SingleQubitGate):
    name = "RY"
    
    def __init__(self, pos: int, paras: float = 0.):
        ParametricGate.__init__(self, pos, paras=paras)

    @property
    def matrix(self):
        return ry_mat(self.paras)


class RZGate(ParametricGate, SingleQubitGate):
    name = "RZ"
    
    def __init__(self, pos: int, paras: float = 0.):
        ParametricGate.__init__(self, pos, paras=paras)
        
    @property
    def matrix(self):
        return rz_mat(self.paras)


class RXXGate(ParametricGate):
    name = "RXX"

    def __init__(self, q1: int, q2: int, paras: float = 0.):
        ParametricGate.__init__(self, [q1, q2], paras=paras)

    @property
    def matrix(self):
        return rxx_mat(self.paras)
    
    @property
    def named_pos(self) -> Dict:
        return {'pos': self.pos}


class RYYGate(ParametricGate):
    name = "RYY"

    def __init__(self, q1: int, q2: int, paras: float = 0.):
        ParametricGate.__init__(self, [q1, q2], paras=paras)

    @property
    def matrix(self):
        return ryy_mat(self.paras)

    @property
    def named_pos(self) -> Dict:
        return {'pos': self.pos}


class RZZGate(ParametricGate):
    name = "RZZ"

    def __init__(self, q1: int, q2: int, paras: float = 0.):
        ParametricGate.__init__(self, [q1, q2], paras=paras)

    @property
    def matrix(self):
        return rzz_mat(self.paras)

    @property
    def named_pos(self) -> Dict:
        return {'pos': self.pos}


QuantumGate.register_gate(RXGate)
QuantumGate.register_gate(RYGate)
QuantumGate.register_gate(RZGate)
QuantumGate.register_gate(RXXGate)
QuantumGate.register_gate(RYYGate)
QuantumGate.register_gate(RZZGate)
