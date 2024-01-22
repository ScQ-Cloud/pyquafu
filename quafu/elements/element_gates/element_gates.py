#  (C) Copyright 2023 Beijing Academy of Quantum Information Sciences
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

""" Actual classes of quantum gates.

The ideal about structuring is roughly "Gate + Ctrl-Gate".


"""

from typing import Dict
from abc import ABC

import quafu.elements.matrices as mat
from quafu.elements.quantum_gate import QuantumGate, ControlledGate
from quafu.elements.quantum_gate import MultiQubitGate, FixedGate, SingleQubitGate, ParametricGate


# # # # # # # # # # # # # # # Internal helper classes # # # # # # # # # # # # # # #

class _C11Gate(ControlledGate, ABC):
    ct_dims = (1, 1, 2)


# # # # # # # # # # # # # # # Paulis # # # # # # # # # # # # # # #
@QuantumGate.register('id')
class IdGate(FixedGate, SingleQubitGate):
    name = "Id"
    matrix = mat.XMatrix

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)


@QuantumGate.register('x')
class XGate(FixedGate, SingleQubitGate):
    name = "X"
    matrix = mat.XMatrix

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)


@QuantumGate.register('y')
class YGate(FixedGate, SingleQubitGate):
    name = "Y"
    matrix = mat.YMatrix

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)


@QuantumGate.register('z')
class ZGate(FixedGate, SingleQubitGate):
    name = "Z"
    matrix = mat.ZMatrix

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)


# # # # # # # # # # # # # # # Sqrt Paulis # # # # # # # # # # # # # # #
@QuantumGate.register('sx')
class SXGate(FixedGate, SingleQubitGate):
    name = "SX"
    matrix = mat.SXMatrix

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)


@QuantumGate.register('sxdg')
class SXdgGate(FixedGate, SingleQubitGate):
    name = "SXdg"
    matrix = mat.SXMatrix.conj().T

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)
        self.symbol = "√X†"


@QuantumGate.register('sy')
class SYGate(FixedGate, SingleQubitGate):
    name = "SY"
    matrix = mat.SYMatrix

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)
        self.symbol = "√Y"

    def to_qasm(self):
        return "ry(pi/2) q[%d];" % self.pos


@QuantumGate.register('sydg')
class SYdgGate(FixedGate, SingleQubitGate):
    name = "SYdg"
    matrix = mat.SYMatrix.conj().T

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)
        self.symbol = "√Y†"

    def to_qasm(self):
        return "ry(-pi/2) q[%d]" % self.pos


@QuantumGate.register('s')
class SGate(SingleQubitGate, FixedGate):
    """SZ"""
    name = "S"
    matrix = mat.SMatrix

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)


@QuantumGate.register('sdg')
class SdgGate(SingleQubitGate, FixedGate):
    name = "Sdg"
    matrix = mat.SMatrix.conj().T

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)


@QuantumGate.register('t')
class TGate(SingleQubitGate, FixedGate):
    name = "T"
    matrix = mat.TMatrix

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)


@QuantumGate.register('tdg')
class TdgGate(SingleQubitGate, FixedGate):
    name = "Tdg"
    matrix = mat.TMatrix.conj().T

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)


# # # # # # # # # # # # # Pauli Linear Combinations # # # # # # # # # # # # #
@QuantumGate.register('h')
class HGate(SingleQubitGate, FixedGate):
    """ (X+Z)/sqrt(2) """
    name = "H"
    matrix = mat.HMatrix

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)


@QuantumGate.register('w')
class WGate(FixedGate, SingleQubitGate):
    """ (X+Y)/sqrt(2) """
    name = "W"
    matrix = mat.WMatrix

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)
        self.symbol = "W"

    def to_qasm(self):
        return "rz(-pi/4) q[%d];\nrx(pi) q[%d];\nrz(pi/4) q[%d]" % (
            self.pos,
            self.pos,
            self.pos,
        )


@QuantumGate.register('sw')
class SWGate(FixedGate, SingleQubitGate):
    name = "SW"
    matrix = mat.SWMatrix

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)
        self.symbol = "√W"

    def to_qasm(self):
        return "rz(-pi/4) q[%d];\nrx(pi/2) q[%d];\nrz(pi/4) q[%d]" % (
            self.pos,
            self.pos,
            self.pos,
        )


@QuantumGate.register('swdg')
class SWdgGate(FixedGate, SingleQubitGate):
    name = "SWdg"
    matrix = mat.SWMatrix

    def __init__(self, pos: int):
        FixedGate.__init__(self, pos)
        self.symbol = "√W†"

    def to_qasm(self):
        return "rz(-pi/4) q[%d];\nrx(-pi/2) q[%d];\nrz(pi/4) q[%d]" % (
            self.pos,
            self.pos,
            self.pos,
        )


# # # # # # # # # # # # # Rotations # # # # # # # # # # # # #
@QuantumGate.register('rx')
class RXGate(ParametricGate, SingleQubitGate):
    name = "RX"

    def __init__(self, pos: int, paras: float = 0.):
        ParametricGate.__init__(self, pos, paras=paras)

    @property
    def matrix(self):
        return mat.rx_mat(self.paras)


@QuantumGate.register('ry')
class RYGate(ParametricGate, SingleQubitGate):
    name = "RY"

    def __init__(self, pos: int, paras: float = 0.):
        ParametricGate.__init__(self, pos, paras=paras)

    @property
    def matrix(self):
        return mat.ry_mat(self.paras)


@QuantumGate.register('rz')
class RZGate(ParametricGate, SingleQubitGate):
    name = "RZ"

    def __init__(self, pos: int, paras: float = 0.):
        ParametricGate.__init__(self, pos, paras=paras)

    @property
    def matrix(self):
        return mat.rz_mat(self.paras)


@SingleQubitGate.register(name='p')
class PhaseGate(SingleQubitGate):
    """Ally of rz gate, with a different name and global phase."""
    name = "P"

    def __init__(self, pos: int, paras: float = 0.0):
        super().__init__(pos, paras=paras)

    @property
    def matrix(self):
        return mat.pmatrix(self.paras)

    @property
    def named_paras(self) -> Dict:
        return {'phase': self.paras}


@QuantumGate.register('rxx')
class RXXGate(ParametricGate):
    name = "RXX"

    def __init__(self, q1: int, q2: int, paras: float = 0.):
        ParametricGate.__init__(self, [q1, q2], paras=paras)

    @property
    def matrix(self):
        return mat.rxx_mat(self.paras)

    @property
    def named_pos(self) -> Dict:
        return {'pos': self.pos}


@QuantumGate.register('ryy')
class RYYGate(ParametricGate):
    name = "RYY"

    def __init__(self, q1: int, q2: int, paras: float = 0.):
        ParametricGate.__init__(self, [q1, q2], paras=paras)

    @property
    def matrix(self):
        return mat.ryy_mat(self.paras)

    @property
    def named_pos(self) -> Dict:
        return {'pos': self.pos}


@QuantumGate.register('rzz')
class RZZGate(ParametricGate):
    name = "RZZ"

    def __init__(self, q1: int, q2: int, paras: float = 0.):
        ParametricGate.__init__(self, [q1, q2], paras=paras)

    @property
    def matrix(self):
        return mat.rzz_mat(self.paras)

    @property
    def named_pos(self) -> Dict:
        return {'pos': self.pos}


# # # # # # # # # # # # # Ctrl-Paulis # # # # # # # # # # # # #
# TODO: implement these by using the META of CtrlGate
@QuantumGate.register('cx')
class CXGate(_C11Gate, FixedGate):
    name = "CX"

    def __init__(self, ctrl: int, targ: int):
        _C11Gate.__init__(self, "X", [ctrl], [targ], None, tar_matrix=mat.XMatrix)
        self.symbol = "+"


@QuantumGate.register('cy')
class CYGate(_C11Gate, FixedGate):
    name = "CY"

    def __init__(self, ctrl: int, targ: int):
        _C11Gate.__init__(self, "Y", [ctrl], [targ], None, tar_matrix=mat.YMatrix)


@QuantumGate.register('cz')
class CZGate(_C11Gate, FixedGate):
    name = "CZ"

    def __init__(self, ctrl: int, targ: int):
        _C11Gate.__init__(self, "Z", [ctrl], [targ], None, tar_matrix=mat.ZMatrix)


@QuantumGate.register('cs')
class CSGate(_C11Gate, FixedGate):
    name = "CS"

    def __init__(self, ctrl: int, targ: int):
        _C11Gate.__init__(self, "S", [ctrl], [targ], None, tar_matrix=mat.SMatrix)

    def to_qasm(self):
        return "cp(pi/2) " + "q[%d],q[%d]" % (self.pos[0], self.pos[1])


@QuantumGate.register('ct')
class CTGate(_C11Gate, FixedGate):
    name = "CT"

    def __init__(self, ctrl: int, targ: int):
        _C11Gate.__init__(self, "T", [ctrl], [targ], None, tar_matrix=mat.TMatrix)

    def to_qasm(self):
        return "cp(pi/4) " + "q[%d],q[%d]" % (self.pos[0], self.pos[1])


# # # # # # # # # # # # # Ctrl-Rotation # # # # # # # # # # # # #
# note: this is the only ctrl-gate that is not a FixedGate
@QuantumGate.register('cp')
class CPGate(_C11Gate):
    name = "CP"

    def __init__(self, ctrl: int, targ: int, paras):
        _C11Gate.__init__(self, "P", [ctrl], [targ], paras, tar_matrix=mat.pmatrix)

    @property
    def named_paras(self) -> Dict:
        return {'theta': self.paras}


# # # # # # # # # # # # # MultiCtrl-Paulis # # # # # # # # # # # # #
@QuantumGate.register('mcx')
class MCXGate(ControlledGate, FixedGate):
    name = "MCX"

    def __init__(self, ctrls, targ: int):
        ControlledGate.__init__(self, "X", ctrls, [targ], None, tar_matrix=mat.XMatrix)


@QuantumGate.register('mcy')
class MCYGate(ControlledGate, FixedGate):
    name = "MCY"

    def __init__(self, ctrls, targ: int):
        ControlledGate.__init__(self, "Y", ctrls, [targ], None, tar_matrix=mat.YMatrix)


@QuantumGate.register('mcz')
class MCZGate(ControlledGate, FixedGate):
    name = "MCZ"

    def __init__(self, ctrls, targ: int):
        ControlledGate.__init__(self, "Z", ctrls, [targ], None, tar_matrix=mat.ZMatrix)


@QuantumGate.register('ccx')
class ToffoliGate(ControlledGate, FixedGate):
    name = "CCX"

    def __init__(self, ctrl1: int, ctrl2: int, targ: int):
        ControlledGate.__init__(self, "X", [ctrl1, ctrl2], [targ], None, tar_matrix=mat.XMatrix)


# # # # # # # # # # # # # SWAPs # # # # # # # # # # # # #
@QuantumGate.register('swap')
class SwapGate(FixedGate, MultiQubitGate):
    name = "SWAP"
    matrix = mat.SwapMatrix

    def __init__(self, q1: int, q2: int):
        super().__init__([q1, q2])
        self.symbol = "x"

    def get_targ_matrix(self, reverse_order=False):
        return self.matrix

    @property
    def named_pos(self) -> Dict:
        return {'pos': self.pos}


@QuantumGate.register('iswap')
class ISwapGate(FixedGate, MultiQubitGate):
    name = "iSWAP"
    matrix = mat.ISwapMatrix

    def __init__(self, q1: int, q2: int):
        super().__init__([q1, q2])
        self.symbol = "(x)"

    def get_targ_matrix(self, reverse_order=False):
        return self.matrix

    @property
    def named_pos(self) -> Dict:
        return {'pos': self.pos}


@QuantumGate.register('cswap')
class FredkinGate(ControlledGate, FixedGate):
    name = "CSWAP"

    def __init__(self, ctrl: int, targ1: int, targ2: int):
        ControlledGate.__init__(self, "SWAP", [ctrl], [targ1, targ2], None, tar_matrix=mat.SwapMatrix)


QuantumGate.register_gate(ToffoliGate, 'toffoli')
QuantumGate.register_gate(FredkinGate, 'fredkin')
