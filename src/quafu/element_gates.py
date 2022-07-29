# This is the file for  concrete element quantum gates
from .quantum_element import FixedSingleQubitGate, ParaSingleQubitGate, FixedTwoQubitGate, ParaTwoQubitGate, \
    ControlGate
import numpy as np
from typing import Union, List


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
    def __init__(self, pos: List[int, int]):
        super().__init__("iSWAP", pos, matrix=np.array([[1., 0., 0., 0.],
                                                        [0., 0., 1.j, 0.],
                                                        [0., 1.j, 0., 0.],
                                                        [0., 0., 0., 1.]], dtype=complex))

    def to_QLisp(self):
        raise ValueError(
            "The BAQIS backend does not support iSWAP gate currently, please use the IOP backend.")


class SwapGate(FixedTwoQubitGate):
    def __init__(self, pos: List[int, int]):
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
    def __init__(self, pos: List[int, int]):
        super().__init__("CX", pos[0], pos[1], matrix=_cxmatrix(reverse=bool(pos[0] > pos[1])))

    def to_QLisp(self):
        return ("Cnot", ("Q%d" % self.ctrl, "Q%d" % self.targ))


class CYGate(ControlGate):
    def __init__(self, pos: List[int, int]):
        super().__init__("CY", pos[0], pos[1], matrix=_cymatrix(reverse=bool(pos[0] > pos[1])))


class CZGate(ControlGate):
    def __init__(self, pos: List[int, int]):
        super().__init__("CZ", pos[0], pos[1], matrix=np.array([[1., 0., 0., 0.],
                                                                [0., 1., 0., 0.],
                                                                [0., 0., 1., 0.],
                                                                [0., 0., 0., -1.]], dtype=complex))


class FsimGate(ParaTwoQubitGate):
    def __init__(self, pos: List[int, int], paras):
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
