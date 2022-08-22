# This is the file for  concrete element quantum gates
from .quantum_element import FixedSingleQubitGate, ParaSingleQubitGate, FixedTwoQubitGate, ParaTwoQubitGate, \
    ControlGate, FixedMultiQubitGate, ParaMultiQubitGate
import numpy as np
from typing import Union, List
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


def _cxmatrix(reverse=False):
    if reverse:
        return np.array([[1., 0., 0., 0.],
                         [0., 0., 0., 1.],
                         [0., 0., 1., 0.],
                         [0., 1., 0., 0.]], dtype=complex)
    else:
        return np.array([[1., 0., 0., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 0., 1.],
                         [0., 0., 1., 0.]], dtype=complex)


def _cymatrix(reverse=False):
    if reverse:
        return np.array([[1., 0., 0., 0.],
                         [0., 0., 0., -1.j],
                         [0., 0., 1., 0.],
                         [0., 1.j, 0., 0.]], dtype=complex)
    else:
        return np.array([[1., 0., 0., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 0., -1.j],
                         [0., 0., 1.j, 0.]], dtype=complex)


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

class WGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("W", pos, matrix=np.zeros((2, 2), dtype=complex))
        self.matrix = (XGate(0).matrix + YGate(0).matrix)/np.sqrt(2)

class SXGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("SX", pos, matrix=np.zeros((2, 2), dtype=complex))
        self.matrix = sqrtm(XGate(0).matrix)

class SYGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("SY", pos, matrix=np.zeros((2, 2), dtype=complex))
        self.matrix = sqrtm(YGate(0).matrix)

class SWGate(FixedSingleQubitGate):
    def __init__(self, pos: int):
        super().__init__("SW", pos, matrix=np.zeros((2, 2), dtype=complex))
        self.matrix = sqrtm(WGate(0).matrix)

class RXGate(ParaSingleQubitGate):
    def __init__(self, pos: int, paras):
        super().__init__("RX", pos, paras, matrix=_rxmatrix)

    def to_QLisp(self):
        return (("Rx", self.paras), "Q%d" % self.pos)


class RYGate(ParaSingleQubitGate):
    def __init__(self, pos: int, paras):
        super().__init__("RY", pos, paras, matrix=_rymatrix)

    def to_QLisp(self):
        return (("Ry", self.paras), "Q%d" % self.pos)


class RZGate(ParaSingleQubitGate):
    def __init__(self, pos: int, paras):
        super().__init__("RZ", pos, paras, matrix=_rzmatrix)

    def to_QLisp(self):
        return (("Rz", self.paras), "Q%d" % self.pos)


class iSwapGate(FixedTwoQubitGate):
    def __init__(self, pos: List[int]):
        super().__init__("iSWAP", pos, matrix=np.array([[1., 0., 0., 0.],
                                                        [0., 0., 1.j, 0.],
                                                        [0., 1.j, 0., 0.],
                                                        [0., 0., 0., 1.]], dtype=complex))

    def to_QLisp(self):
        raise ValueError(
            "The BAQIS backend does not support iSWAP gate currently, please use the IOP backend.")


class SwapGate(FixedTwoQubitGate):
    def __init__(self, pos: List[int]):
        super().__init__("SWAP", pos, matrix=np.array([[1., 0., 0., 0.],
                                                       [0., 0., 1., 0.],
                                                       [0., 1., 0., 0.],
                                                       [0., 0., 0., 1.]], dtype=complex))

    def to_QLisp(self):
        raise ValueError(
            "The BAQIS backend does not support SWAP gate currently, please use the IOP backend.")

    def to_nodes(self):
        raise ValueError(
            "The IOP backend does not support SWAP gate currently, please use the BAQIS backend.")

    def to_IOP(self):
        raise ValueError(
            "The IOP backend does not support SWAP gate currently, please use the BAQIS backend.")


class CXGate(ControlGate):
    def __init__(self, pos: List[int]):
        super().__init__("CX", pos[0], pos[1], matrix=_cxmatrix(reverse=bool(pos[0] > pos[1])))
        self.targ_name = "X"

    def to_QLisp(self):
        return ("Cnot", ("Q%d" % self.ctrl, "Q%d" % self.targ))

class CYGate(ControlGate):
    def __init__(self, pos: List[int]):
        super().__init__("CY", pos[0], pos[1], matrix=_cymatrix(reverse=bool(pos[0] > pos[1])))
        self.targ_name = "Y"

class CZGate(ControlGate):
    def __init__(self, pos: List[int]):
        super().__init__("CZ", pos[0], pos[1], matrix=np.array([[1., 0., 0., 0.],
                                                                [0., 1., 0., 0.],
                                                                [0., 0., 1., 0.],
                                                                [0., 0., 0., -1.]], dtype=complex))
        
        self.targ_name = "Z"

class CSGate(ControlGate):
    def __init__(self, pos: List[int]):
        super().__init__("CS", pos[0], pos[1], matrix=np.array([[1., 0., 0., 0.],
                                                                [0., 1., 0., 0.],
                                                                [0., 0., 1., 0.],
                                                                [0., 0., 0., 1.j]], dtype=complex))
        self.targ_name = "S"


class CTGate(ControlGate):
    def __init__(self, pos: List[int]):
        super().__init__("CS", pos[0], pos[1], matrix=np.array([[1., 0., 0., 0.],
                                                                [0., 1., 0., 0.],
                                                                [0., 0., 1., 0.],
                                                                [0., 0., 0., np.exp(1.j*np.pi/4)]], dtype=complex))
        self.targ_name = "T"



class ToffoliGate(FixedMultiQubitGate):
    def __init__(self, pos):
        super().__init__("CCX", pos, np.zeros([8, 8]))
        self.ctrls = pos[0:2]
        self.targs = [pos[2]]
        self.targ_names = ["X"]

        for i in range(6):
            self.matrix[i, i] = 1.
        self.matrix[6, 7] = 1.
        self.matrix[7, 6] = 1.

        inds = np.argsort(pos)
        inds = np.concatenate([inds, inds+3])
        tensorm = self.matrix.reshape([2, 2, 2, 2, 2, 2])
        self.matrix = np.transpose(tensorm, inds).reshape([8, 8])


class FredkinGate(FixedMultiQubitGate):
    def __init__(self, pos):
        super().__init__("CSwap", pos, np.zeros([8, 8]))
        self.ctrls = [pos[0]]
        self.targs = pos[1:]
        self.targ_names = ["SWAP", "SWAP"]

        for i in range(5):
            self.matrix[i, i] = 1.
        
        self.matrix[5, 6] = 1.
        self.matrix[6, 5] = 1.
        self.matrix[7, 7] = 1.

        inds = np.argsort(pos)
        inds = np.concatenate([inds, inds+3])
        tensorm = self.matrix.reshape([2, 2, 2, 2, 2, 2])
        self.matrix = np.transpose(tensorm, inds).reshape([8, 8])


class FsimGate(ParaTwoQubitGate):
    def __init__(self, pos: List[int], paras):
        super().__init__("fSim", pos, paras)
        self.__theta = paras[0]
        self.__phi = paras[1]

    def to_nodes(self):
        raise ValueError(
            "The IOP backend does not support fSim gate currently, please use the BAQIS backend.")

    def to_IOP(self):
        raise ValueError(
            "The IOP backend does not support fSim gate currently, please use the BAQIS backend.")

def main():
    res = iSwapGate([0, 1])
    print(str(res))
    return res


if __name__ == '__main__':
    main()
