from abc import ABC, abstractmethod
from typing import Callable, List, Union, Iterable

import numpy as np

from quafu.elements.quantum_element.instruction import Instruction, PosType


def reorder_matrix(matrix: np.ndarray, pos: List):
    """Reorder the input sorted matrix to the pos order """
    qnum = len(pos)
    dim = 2 ** qnum
    inds = np.argsort(pos)
    inds = np.concatenate([inds, inds + qnum])
    tensorm = matrix.reshape([2] * 2 * qnum)
    return np.transpose(tensorm, inds).reshape([dim, dim])


class QuantumGate(Instruction):
    def __init__(self,
                 pos: PosType,
                 paras: Union[float, List[float]] = None,
                 ):
        self.pos = pos
        self.paras = paras

        if paras:
            if isinstance(paras, Iterable):
                self.symbol = "%s(" % self.name + ",".join(["%.3f" % para for para in self.paras]) + ")"
            else:
                self.symbol = "%s(%.3f)" % (self.name, paras)
        else:
            self.symbol = "%s" % self.name

    @property
    @abstractmethod
    def matrix(self):
        raise NotImplementedError("Matrix is not implemented for %s" % self.__class__.__name__ +
                                  ", this should never happen.")

    # @property
    # def pos(self):
    #     return self.__pos
    #
    # @pos.setter
    # def pos(self, _pos):
    #     self.__pos = _pos
    #
    # @property
    # def paras(self):
    #     return self.__paras
    #
    # @paras.setter
    # def paras(self, _paras):
    #     self.__paras = _paras

    def __str__(self):
        properties_names = ['pos', 'paras', 'matrix']
        properties_values = [getattr(self, x) for x in properties_names]
        return "%s:\n%s" % (self.__class__.__name__, '\n'.join(
            [f"{x} = {repr(properties_values[i])}" for i, x in enumerate(properties_names)]))

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def to_qasm(self):
        qstr = "%s" % self.name.lower()

        if self.paras:
            if isinstance(self.paras, Iterable):
                qstr += "(" + ",".join(["%s" % para for para in self.paras]) + ")"
            else:
                qstr += "(%s)" % self.paras
        qstr += " "
        if isinstance(self.pos, Iterable):
            qstr += ",".join(["q[%d]" % p for p in self.pos])
        else:
            qstr += "q[%d]" % self.pos

        return qstr


class SingleQubitGate(QuantumGate, ABC):
    def __init__(self, pos: int, paras: float = None):
        super().__init__(pos, paras=paras)

    # @property
    # def matrix(self):
    #     return self._matrix
    #
    # @matrix.setter
    # def matrix(self, matrix):
    #     if isinstance(matrix, (np.ndarray, List)):
    #         if np.shape(matrix) == (2, 2):
    #             self._matrix = np.asarray(matrix, dtype=complex)
    #         else:
    #             raise ValueError(f'`{self.__class__.__name__}.matrix.shape` must be (2, 2)')
    #     elif isinstance(matrix, type(None)):
    #         self._matrix = matrix
    #     else:
    #         raise TypeError("Unsupported `matrix` type")

    def get_targ_matrix(self, reverse_order=False):
        return self.matrix


class MultiQubitGate(QuantumGate, ABC):
    def __init__(self, pos: List[int], paras: float = None):
        super().__init__(pos, paras)
        self._targ_matrix = self.matrix

    # @property
    # def matrix(self):
    #     return self._matrix
    #
    # @matrix.setter
    # def matrix(self, matrix):
    #     if isinstance(matrix, np.ndarray):
    #         self._matrix = matrix
    #     elif isinstance(matrix, List):
    #         self._matrix = np.array(matrix, dtype=complex)
    #     else:
    #         raise TypeError("Unsupported `matrix` type")
    #
    #     self._matrix = reorder_matrix(self._matrix, self.pos)

    def get_targ_matrix(self, reverse_order=False):
        targ_matrix = self._targ_matrix
        if reverse_order and (len(self.pos) > 1):
            qnum = len(self.pos)
            order = np.array(range(len(self.pos))[::-1])
            order = np.concatenate([order, order + qnum])
            dim = 2 ** qnum
            tensorm = targ_matrix.reshape([2] * 2 * qnum)
            targ_matrix = np.transpose(tensorm, order).reshape([dim, dim])

        return targ_matrix


class ParaSingleQubitGate(SingleQubitGate, ABC):
    def __init__(self, pos, paras: float):
        if paras is None:
            raise ValueError("`paras` can not be None for ParaSingleQubitGate")
        elif not isinstance(paras, float):
            raise TypeError("`paras` must be float for ParaSingleQubitGate")
        super().__init__(pos, paras=paras)

    # @property
    # def matrix(self):
    #     return self._matrix
    #
    # @matrix.setter
    # def matrix(self, matrix):
    #     if isinstance(matrix, Callable):
    #         self._matrix = matrix(self.paras)
    #     elif isinstance(matrix, (np.ndarray, List)):
    #         if np.shape(matrix) == (2, 2):
    #             self._matrix = np.asarray(matrix, dtype=complex)
    #         else:
    #             raise ValueError(f'`{self.__class__.__name__}.matrix.shape` must be (2, 2)')
    #     elif isinstance(matrix, type(None)):
    #         self._matrix = matrix
    #     else:
    #         raise TypeError("Unsupported `matrix` type")


class FixedSingleQubitGate(SingleQubitGate, ABC):
    def __init__(self, pos):
        super().__init__(pos=pos, paras=None)


class FixedMultiQubitGate(MultiQubitGate, ABC):
    def __init__(self, pos: List[int]):
        super().__init__(pos=pos, paras=None)


class ControlledGate(MultiQubitGate, ABC):
    """ Controlled gate class, where the matrix act non-trivaly on target qubits"""

    def __init__(self, targe_name, ctrls: List[int], targs: List[int], paras, matrix):
        self.ctrls = ctrls
        self.targs = targs
        self.targ_name = targe_name
        super().__init__(ctrls + targs, paras)
        self._targ_matrix = matrix

        if paras:
            if isinstance(paras, Iterable):
                self.symbol = "%s(" % self.targ_name + ",".join(["%.3f" % para for para in self.paras]) + ")"
            else:
                self.symbol = "%s(%.3f)" % (self.targ_name, paras)
        else:
            self.symbol = "%s" % self.targ_name

    # TODO

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: Union[np.ndarray, Callable]):
        targ_dim = 2 ** (len(self.targs))
        qnum = len(self.pos)
        dim = 2 ** (qnum)
        if isinstance(matrix, Callable):
            matrix = matrix(self.paras)

        if matrix.shape[0] != targ_dim:
            raise ValueError("Dimension dismatch")
        else:
            self._matrix = np.zeros((dim, dim), dtype=complex)
            control_dim = 2 ** len(self.pos) - targ_dim
            for i in range(control_dim):
                self._matrix[i, i] = 1.

            self._matrix[control_dim:, control_dim:] = matrix
            self._matrix = reorder_matrix(self._matrix, self.pos)
            # self._targ_matrix = reorder_matrix(matrix, self.targs)

    def get_targ_matrix(self, reverse_order=False):
        targ_matrix = self._targ_matrix
        if reverse_order and (len(self.targs) > 1):
            qnum = len(self.targs)
            order = np.array(range(len(self.targs))[::-1])
            order = np.concatenate([order, order + qnum])
            dim = 2 ** qnum
            tensorm = targ_matrix.reshape([2] * 2 * qnum)
            targ_matrix = np.transpose(tensorm, order).reshape([dim, dim])

        return targ_matrix


class ParaMultiQubitGate(MultiQubitGate, ABC):
    def __init__(self, pos, paras):
        if paras is None:
            raise ValueError("`paras` can not be None for ParaMultiQubitGate")
        super().__init__(pos, paras)

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        if isinstance(matrix, Callable):
            self._matrix = matrix(self.paras)
            self._matrix = reorder_matrix(self._matrix, self.pos)
        elif isinstance(matrix, (np.ndarray, List)):
            self._matrix = matrix
            self._matrix = reorder_matrix(self._matrix, self.pos)
        else:
            raise TypeError("Unsupported `matrix` type")


class ControlledU(ControlledGate):
    """ Controlled gate class, where the matrix act non-trivially on target qubits"""

    name = 'cu'

    def __init__(self, ctrls: List[int], U: Union[SingleQubitGate, MultiQubitGate]):
        self.targ_gate = U
        targs = U.pos
        if isinstance(targs, int):
            targs = [targs]

        super().__init__(U.name, ctrls, targs, U.paras, matrix=self.targ_gate.get_targ_matrix())

    def get_targ_matrix(self, reverse_order=False):
        return self.targ_gate.get_targ_matrix(reverse_order)
