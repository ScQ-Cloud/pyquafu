# This is the file for  concrete element quantum gates
from .quantum_element import ControlledGate, FixedSingleQubitGate, ParaMultiQubitGate, ParaSingleQubitGate,\
    ControlledGate, FixedMultiQubitGate
import numpy as np
from typing import List
from scipy.linalg import sqrtm


def _u2matrix(_phi=0., _lambda=0.):
    "OpenQASM 3.0 specification"
    return np.array([[1., np.exp(-1.j * _lambda)],
                     [np.exp(1.j * _phi), np.exp((_phi + _lambda) * 1.j)]], dtype=complex)


def _u3matrix(_theta=0., _phi=0., _lambda=0.):
    "OpenQASM 3.0 specification"
    return np.array([[np.cos(0.5 * _theta), -np.exp(_lambda * 1.j) * np.sin(0.5 * _theta)],
                     [np.exp(_phi * 1.j) * np.sin(0.5 * _theta),
                      np.exp((_phi + _lambda) * 1.j) * np.cos(0.5 * _theta)]], dtype=complex)


def _rxmatrix(theta):
    return np.array([[np.cos(0.5 * theta), -1.j * np.sin(0.5 * theta)],
                     [-1.j * np.sin(0.5 * theta), np.cos(0.5 * theta)]], dtype=complex)


def _rymatrix(theta):
    return np.array([[np.cos(0.5 * theta), - np.sin(0.5 * theta)],
                     [np.sin(0.5 * theta), np.cos(0.5 * theta)]], dtype=complex)


def _rzmatrix(theta):
    return np.array([[np.exp(-0.5j * theta), 0.],
                     [0., np.exp(0.5j * theta)]], dtype=complex)

def _pmatrix(labda):
    return np.array([[1, 0], 
                     [0, np.exp(1j*labda)]] ,dtype=complex)

def _rxxmatrix(theta):
    """Unitary evolution of XX interaction"""
    return np.array([[np.cos(theta/2), 0, 0, -1j*np.sin(theta/2)],
                     [0, np.cos(theta/2), -1j*np.sin(theta/2), 0],
                     [0, -1j*np.sin(theta/2), np.cos(theta/2), 0],
                     [-1j*np.sin(theta/2), 0, 0, np.cos(theta/2)]
                    ])

def _ryymatrix(theta):
    """ Unitary evolution of YY interaction"""
    return np.array([[np.cos(theta/2), 0, 0, 1j*np.sin(theta/2)],
                     [0, np.cos(theta/2), -1j*np.sin(theta/2), 0],
                     [0, -1j*np.sin(theta/2), np.cos(theta/2), 0],
                     [1j*np.sin(theta/2), 0, 0, np.cos(theta/2)]
                    ])

def _rzzmatrix(theta):
    return np.array([[np.exp(-1j*theta/2), 0, 0, 0],
                     [0, np.exp(1j*theta/2), 0, 0],
                     [0, 0, np.exp(1j*theta/2), 0],
                     [0, 0, 0, np.exp(-1j*theta/2)]
                    ])

class IdGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("Id", pos, matrix = np.array([[1., 0.], 
                                                       [0., 1.]], dtype=complex))

class HGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("H", pos, matrix=1 / np.sqrt(2) * np.array([[1., 1.],
                                                                     [1., -1.]], dtype=complex))


class XGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("X", pos, matrix=np.array([[0., 1.],
                                                    [1., 0.]], dtype=complex))


class YGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("Y", pos, matrix=np.array([[0., -1.j],
                                                    [1.j, 0.]], dtype=complex))


class ZGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("Z", pos, matrix=np.array([[1., 0.],
                                                    [0., -1.]], dtype=complex))

class SGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("S", pos, matrix=np.array([[1., 0.],
                                                    [0., 1.j]], dtype=complex))

class SdgGate(FixedSingleQubitGate):
    def __init__(sell, pos: int):
        super().__init__("Sdg", pos, matrix = np.array([[1., 0.],
                                                    [0., -1.j]], dtype=complex))

class TGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("T", pos, matrix=np.array([[1., 0.],
                                                    [0., np.exp(1.j*np.pi/4)]], dtype=complex))

class TdgGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("Tdg", pos, matrix=np.array([[1., 0.],
                                                    [0, np.exp(-1.j*np.pi/4)]] ,dtype=complex))

class WGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("W", pos, matrix=np.zeros((2, 2), dtype=complex))
        self.matrix = (XGate(0).matrix + YGate(0).matrix)/np.sqrt(2)

class SXGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("SX", pos, matrix=np.zeros((2, 2), dtype=complex))
        self.matrix = sqrtm(XGate(0).matrix)
        self.symbol = "√X"

class SYGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("SY", pos, matrix=np.zeros((2, 2), dtype=complex))
        self.matrix = sqrtm(YGate(0).matrix)
        self.symbol = "√Y"

class SWGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("SW", pos, matrix=np.zeros((2, 2), dtype=complex))
        self.matrix = sqrtm(WGate(0).matrix)
        self.symbol = "√W"

class RXGate(ParaSingleQubitGate):
    def __init__(self, pos: int, paras):
        super().__init__("RX", pos, paras, matrix=_rxmatrix)


class RYGate(ParaSingleQubitGate):
    def __init__(self, pos: int, paras):
        super().__init__("RY", pos, paras, matrix=_rymatrix)


class RZGate(ParaSingleQubitGate):
    def __init__(self, pos: int, paras):
        super().__init__("RZ", pos, paras, matrix=_rzmatrix)


class PhaseGate(ParaSingleQubitGate):
    def __init__(self, pos: int, paras):
        super().__init__("P", pos, paras, matrix=_pmatrix)
    

class iSwapGate(FixedMultiQubitGate):
    def __init__(self, q1:int, q2:int):
        super().__init__("iSWAP", [q1, q2], matrix=np.array([[1., 0., 0., 0.],
                                                        [0., 0., 1.j, 0.],
                                                        [0., 1.j, 0., 0.],
                                                        [0., 0., 0., 1.]], dtype=complex))

    def get_targ_matrix(self, reverse_order=False):
        return self.matrix


class SwapGate(FixedMultiQubitGate):
    def __init__(self, q1:int, q2:int):
        super().__init__("SWAP", [q1, q2], matrix=np.array([[1., 0., 0., 0.],
                                                       [0., 0., 1., 0.],
                                                       [0., 1., 0., 0.],
                                                       [0., 0., 0., 1.]], dtype=complex))
        self.symbol = "x"

    def get_targ_matrix(self, reverse_order=False):
        return self.matrix

class CXGate(ControlledGate):
    def __init__(self, ctrl:int, targ:int):
        super().__init__("CX", "X", [ctrl], [targ], None, matrix=XGate(0).matrix)
        self.symbol = "+"


class CYGate(ControlledGate):
    def __init__(self, ctrl:int, targ:int):
        super().__init__("CY", "Y", [ctrl], [targ], None, matrix=YGate(0).matrix)


class CZGate(ControlledGate):
    def __init__(self, ctrl:int, targ:int):
        super().__init__("CZ", "Z", [ctrl], [targ], None, matrix=ZGate(0).matrix)

class CSGate(ControlledGate):
    def __init__(self, ctrl:int, targ:int):
        super().__init__("CS", "S", [ctrl], [targ], None, matrix=SGate(0).matrix)


class CTGate(ControlledGate):
    def __init__(self, ctrl:int, targ:int):
        super().__init__("CT", "T", [ctrl], [targ], None, matrix=TGate(0).matrix)

class CPGate(ControlledGate):
    def __init__(self, ctrl:int, targ:int, paras):
        super().__init__("CP", "P", [ctrl], [targ], paras, matrix=PhaseGate(0, paras).matrix)

class ToffoliGate(ControlledGate):
    def __init__(self, ctrl1:int, ctrl2:int, targ:int):
        super().__init__("CCX", "X", [ctrl1, ctrl2], [targ], None, matrix=XGate(0).matrix)

class FredkinGate(ControlledGate):
    def __init__(self, ctrl:int, targ1:int, targ2:int):
        super().__init__("CSWAP", "SWAP", [ctrl], [targ1, targ2], None, matrix=SwapGate(0, 1).matrix)


class RXXGate(ParaMultiQubitGate):
    def __init__(self, q1:int, q2:int, theta):
        super().__init__("RXX", [q1, q2], theta, matrix=_rxxmatrix(theta))
    
    def get_targ_matrix(self, reverse_order=False):
        return self.matrix

class RYYGate(ParaMultiQubitGate):
    def __init__(self, q1:int, q2:int, theta):
        super().__init__("RYY", [q1, q2], theta, matrix=_ryymatrix(theta))

    def get_targ_matrix(self, reverse_order=False):
        return self.matrix

class RZZGate(ParaMultiQubitGate):
    def __init__(self, q1:int, q2:int, theta):
        super().__init__("RZZ", [q1, q2], theta, matrix=_rzzmatrix(theta))

    def get_targ_matrix(self, reverse_order=False):
        return self.matrix

class MCXGate(ControlledGate):
    def __init__(self, ctrls, targ:int):
        super().__init__("MCX", "X", ctrls, [targ], None, matrix=XGate(0).matrix)

class MCYGate(ControlledGate):
    def __init__(self, ctrls, targ:int):
        super().__init__("MCY", "Y", ctrls, [targ], None, matrix=YGate(0).matrix)

class MCZGate(ControlledGate):
    def __init__(self, ctrls, targ:int):
        super().__init__("MCZ", "Z", ctrls, [targ], None, matrix=ZGate(0).matrix)

