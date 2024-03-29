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

from typing import Dict, List
from abc import ABC

import quafu.elements.matrices as mat
from quafu.elements.quantum_gate import QuantumGate, ControlledGate
from ..parameters import ParameterType

# # # # # # # # # # # # # # # Internal helper classes # # # # # # # # # # # # # # #

class _C11Gate(ControlledGate, ABC):
    ct_dims = (1, 1, 2)

def wrap_para(matfunc):
    def wrap_func(paras:List[ParameterType]):
        return matfunc(*paras) 
    return wrap_func

# # # # # # # # # # # # # # # Paulis # # # # # # # # # # # # # # #
@QuantumGate.register()
class IdGate(QuantumGate):
    name = "ID"
    _raw_matrix = mat.IdMatrix
    paras = [] 
    def __init__(self, pos: int):
       self.pos =  [pos]


@QuantumGate.register()
class XGate(QuantumGate):
    name = "X"
    _raw_matrix = mat.XMatrix
    paras = [] 
    def __init__(self, pos: int):
       self.pos =  [pos]

@QuantumGate.register()
class YGate(QuantumGate):
    name = "Y"
    _raw_matrix = mat.YMatrix
    paras = [] 
    def __init__(self, pos: int):
       self.pos =  [pos]

@QuantumGate.register()
class ZGate(QuantumGate):
    name = "Z"
    _raw_matrix = mat.ZMatrix
    paras = [] 
    def __init__(self, pos: int):
       self.pos =  [pos]


# # # # # # # # # # # # # # # Sqrt Paulis # # # # # # # # # # # # # # #
@QuantumGate.register()
class SGate(QuantumGate):
    name = "S"
    _raw_matrix = mat.SMatrix
    paras = [] 
    def __init__(self, pos: int):
       self.pos =  [pos]

@QuantumGate.register()
class SdgGate(QuantumGate):
    name = "Sdg"
    _raw_matrix = mat.SMatrix.conj().T
    paras = [] 
    def __init__(self, pos: int):
       self.pos =  [pos]

@QuantumGate.register()
class TGate(QuantumGate):
    name = "T"
    _raw_matrix = mat.TMatrix
    paras = [] 
    def __init__(self, pos: int):
       self.pos =  [pos]

@QuantumGate.register()
class TdgGate(QuantumGate):
    name = "Tdg"
    _raw_matrix = mat.TMatrix.conj().T
    paras = [] 
    def __init__(self, pos: int):
       self.pos =  [pos]

@QuantumGate.register()
class SXGate(QuantumGate):
    name = "SX"
    _raw_matrix = mat.SXMatrix
    paras = [] 
    def __init__(self, pos: int):
       self.pos =  [pos]

@QuantumGate.register()
class SXdgGate(QuantumGate):
    name = "SXdg"
    _raw_matrix = mat.SXMatrix.conj().T

    paras = [] 
    def __init__(self, pos: int):
       self.pos =  [pos]

@QuantumGate.register()
class SYGate(QuantumGate):
    name = "SY"
    _raw_matrix = mat.SYMatrix
    paras = [] 
    def __init__(self, pos: int):
       self.pos =  [pos]

    def to_qasm(self, with_para):
        return "ry(pi/2) q[%d]" %(self.pos[0])

@QuantumGate.register()
class SYdgGate(QuantumGate):
    name = "SYdg"
    _raw_matrix = mat.SYMatrix.conj().T

    paras = [] 
    def __init__(self, pos: int):
       self.pos =  [pos]


    

# # # # # # # # # # # # # Pauli Linear Combinations # # # # # # # # # # # # #
@QuantumGate.register()
class HGate(QuantumGate):
    name = "H"
    _raw_matrix = mat.HMatrix
    paras = [] 
    def __init__(self, pos: int):
       self.pos =  [pos]

@QuantumGate.register()
class WGate(QuantumGate):
    name = "W"
    _raw_matrix = mat.WMatrix
    paras = [] 
    def __init__(self, pos: int):
       self.pos =  [pos]
    
    def to_qasm(self, with_para):
        q = self.pos[0]
        return "rz(-pi/4) q[%d];\nrx(pi) q[%d];\nrz(pi/4) q[%d]"  %(q, q, q)

@QuantumGate.register()
class SWGate(QuantumGate):
    name = "SW"
    _raw_matrix = mat.SWMatrix
    paras = [] 
    def __init__(self, pos: int):
       self.pos =  [pos]

    def to_qasm(self, with_para):
        q = self.pos[0]
        return "rz(-pi/4) q[%d];\nrx(pi/2) q[%d];\nrz(pi/4) q[%d]"  %(q, q, q)

@QuantumGate.register()
class SWdgGate(QuantumGate):
    name = "SWdg"
    _raw_matrix = mat.SWMatrix.conj().T
    paras = [] 
    def __init__(self, pos: int):
       self.pos =  [pos]

    def to_qasm(self, with_para):
        q = self.pos[0]
        return "rz(-pi/4) q[%d];\nrx(-pi/2) q[%d];\nrz(pi/4) q[%d]"  %(q, q, q)



# # # # # # # # # # # # # Rotations # # # # # # # # # # # # #
@QuantumGate.register()
class RXGate(QuantumGate):
    def __init__(self, pos: int, theta:ParameterType):
        super().__init__("RX", [pos], [theta], wrap_para(mat.rx_mat))

@QuantumGate.register()
class RYGate(QuantumGate):
    def __init__(self, pos: int, theta:ParameterType):
        super().__init__("RY", [pos], [theta],  wrap_para(mat.ry_mat))

@QuantumGate.register()
class RZGate(QuantumGate):
    def __init__(self, pos: int, theta:ParameterType):
        super().__init__("RZ", [pos], [theta], wrap_para(mat.rz_mat))

# @QuantumGate.register()
# class U2(QuantumGate):
#     def __init__(self, pos: int, phi: float, _lambda: float):
#         super().__init__("U2", [pos], [phi, _lambda], u2matrix(phi, _lambda))

@QuantumGate.register()
class U3Gate(QuantumGate):
    def __init__(self, pos: int, theta : ParameterType, phi: ParameterType, _lambda: ParameterType):
        super().__init__("U3", [pos], [theta, phi, _lambda], matrix = wrap_para(mat.u3matrix))

@QuantumGate.register(name='p')
class PhaseGate(QuantumGate):
    def __init__(self, pos: int, _lambda:ParameterType):
        super().__init__("P", [pos], [_lambda], wrap_para(mat.pmatrix))

@QuantumGate.register()
class RXXGate(QuantumGate):
    def __init__(self, q1:int, q2:int, theta:ParameterType):
        super().__init__("RXX", [q1, q2], [theta], wrap_para(mat.rxx_mat))
    
    
@QuantumGate.register()
class RYYGate(QuantumGate):
    def __init__(self, q1:int, q2:int, theta:ParameterType):
        super().__init__("RYY", [q1, q2], [theta], wrap_para(mat.ryy_mat))

    
@QuantumGate.register()
class RZZGate(QuantumGate):
    def __init__(self, q1:int, q2:int, theta:ParameterType):
        super().__init__("RZZ", [q1, q2], [theta], wrap_para(mat.rzz_mat))



#--------------------ControalledGate----------
# TODO: implement using ControllU class
# # # # # # # # # # # # # Ctrl-Paulis # # # # # # # # # # # # #
@QuantumGate.register()
class CXGate(ControlledGate):
    name  = "CX"
    _targ_name = "X"
    _targ_matrix = mat.XMatrix
    _raw_matrix = mat.CXMatrix
    paras = []
    def __init__(self, ctrl:int, targ:int):
        assert ctrl != targ
        self.ctrls  = [ctrl]
        self.targs = [targ]
        self.pos = self.ctrls + self.targs

    @property
    def symbol(self):
        return "+"

@QuantumGate.register()
class CYGate(ControlledGate):
    name  = "CY"
    _targ_name = "Y"
    _targ_matrix = mat.YMatrix
    _raw_matrix = mat.CYMatrix
    paras = []
    def __init__(self, ctrl:int, targ:int):
        assert ctrl != targ
        self.ctrls  = [ctrl]
        self.targs = [targ]
        self.pos = self.ctrls + self.targs

@QuantumGate.register()
class CZGate(ControlledGate):
    name  = "CZ"
    _targ_name = "Z"
    _targ_matrix = mat.ZMatrix
    _raw_matrix = mat.CZMatrix
    paras = []
    def __init__(self, ctrl:int, targ:int):
        assert ctrl != targ
        self.ctrls  = [ctrl]
        self.targs = [targ]
        self.pos = self.ctrls + self.targs

@QuantumGate.register()
class CSGate(ControlledGate):
    name  = "CS"
    _targ_name = "S"
    _targ_matrix = mat.XMatrix
    _raw_matrix = mat.CXMatrix
    paras = []
    def __init__(self, ctrl:int, targ:int):
        assert ctrl != targ
        self.ctrls  = [ctrl]
        self.targs = [targ]
        self.pos = self.ctrls + self.targs

    def to_qasm(self, with_para):
        return "cp(pi/2) " + "q[%d],q[%d]" % (self.pos[0], self.pos[1])
    
@QuantumGate.register()
class CTGate(ControlledGate):
    name  = "CT"
    _targ_name = "T"
    _targ_matrix = mat.TMatrix
    _raw_matrix = mat.CTMatrix
    paras = []
    def __init__(self, ctrl:int, targ:int):
        assert ctrl != targ
        self.ctrls  = [ctrl]
        self.targs = [targ]
        self.pos = self.ctrls + self.targs

    def to_qasm(self, with_para):
        return "cp(pi/4) " + "q[%d],q[%d]" % (self.pos[0], self.pos[1])


# # # # # # # # # # # # # Ctrl-Rotation # # # # # # # # # # # # #
# note: this is the only ctrl-gate that is not a FixedGate
@QuantumGate.register()
class CPGate(ControlledGate):
    def __init__(self, ctrl:int, targ:int, _lambda:ParameterType):
        super().__init__("CP", "P", [ctrl], [targ], [_lambda], wrap_para(mat.pmatrix))



@QuantumGate.register()
class CRXGate(ControlledGate):
    def __init__(self, ctrl:int, targ:int, theta:ParameterType):
        super().__init__("CRX", "RX", [ctrl], [targ], [theta], wrap_para(mat.rx_mat))

@QuantumGate.register()
class CRYGate(ControlledGate):
    def __init__(self, ctrl:int, targ:int, theta:ParameterType):
        super().__init__("CRY", "RY", [ctrl], [targ], [theta], wrap_para(mat.ry_mat))

@QuantumGate.register()
class CRZGate(ControlledGate):
    def __init__(self, ctrl:int, targ:int, theta:ParameterType):
        super().__init__("CRZ", "RZ", [ctrl], [targ], [theta], wrap_para(mat.rz_mat))

# # # # # # # # # # # # # MultiCtrl-Paulis # # # # # # # # # # # # #
@QuantumGate.register()
class MCXGate(ControlledGate):
    def __init__(self, ctrls:List[int], targ:int):
        super().__init__("MCX", "X", ctrls, [targ], [], mat.XMatrix)
    
    @property
    def symbol(self):
        return "+"

@QuantumGate.register()
class MCYGate(ControlledGate):
    def __init__(self, ctrls:List[int], targ:int):
        super().__init__("MCY", "Y", ctrls, [targ], [], mat.YMatrix)

@QuantumGate.register()
class MCZGate(ControlledGate):
    def __init__(self, ctrls:List[int], targ:int):
        super().__init__("MCZ", "Z", ctrls, [targ], [], mat.ZMatrix)

@QuantumGate.register()
class MCRXGate(ControlledGate):
    def __init__(self, ctrls:List[int], targ:int, theta:ParameterType):
        super().__init__("MCRX", "RX", ctrls, [targ], [theta], wrap_para(mat.rx_mat))

@QuantumGate.register()
class MCRYGate(ControlledGate):
    def __init__(self, ctrls:List[int], targ:int, theta:ParameterType):
        super().__init__("MCRY", "RY", ctrls, [targ], [theta], wrap_para(mat.ry_mat))

@QuantumGate.register()
class MCRZGate(ControlledGate):
    def __init__(self, ctrls:List[int], targ:int, theta:ParameterType):
        super().__init__("MCRZ", "RZ", ctrls, [targ], [theta], wrap_para(mat.rz_mat))



@QuantumGate.register()
class CCXGate(ControlledGate):
    def __init__(self, ctrl1:int, ctrl2:int, targ:int):
        super().__init__("CCX", "X", [ctrl1, ctrl2], [targ], [], mat.XMatrix)

@QuantumGate.register()
class CSwapGate(ControlledGate):
    def __init__(self, ctrl:int, targ1:int, targ2:int):
        super().__init__("CSWAP", "SWAP", [ctrl], [targ1, targ2], [], mat.SwapMatrix)

# # # # # # # # # # # # # SWAPs # # # # # # # # # # # # #
@QuantumGate.register()
class SwapGate(QuantumGate):
    name = "SWAP"
    matrix = mat.SwapMatrix
    _raw_matrix = mat.SwapMatrix
    paras = [] 
    def __init__(self, q1:int, q2:int):
       self.pos =  [q1,  q2]
    
    @property
    def symbol(self):
        return "x"


@QuantumGate.register()
class ISwapGate(QuantumGate):
    name = "ISWAP"
    matrix = mat.ISwapMatrix
    _raw_matrix = mat.ISwapMatrix
    paras = [] 
    def __init__(self, q1:int, q2:int):
       self.pos =  [q1,  q2]


QuantumGate.gate_classes['cnot'] = CXGate
QuantumGate.gate_classes['toffoli'] = CCXGate
QuantumGate.gate_classes['fredon'] = CSwapGate
FredkinGate = CSwapGate
ToffoliGate = CCXGate