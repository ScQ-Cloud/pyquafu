from abc import ABC, abstractmethod
from typing import List, Union, Iterable

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
    gate_classes = {}

    def __init__(self,
                 pos: PosType,
                 paras: Union[float, List[float]] = None,
                 ):
        super().__init__(pos, paras)

        if paras is not None:
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

    @classmethod
    def register_gate(cls, subclass, name: str = None):
        assert issubclass(subclass, cls)

        if name is None:
            name = subclass.name
        if name in cls.gate_classes:
            raise ValueError(f"Name {name} already exists.")
        cls.gate_classes[name] = subclass
        Instruction.register_ins(subclass, name)

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

    def get_targ_matrix(self):
        return self.matrix


class MultiQubitGate(QuantumGate, ABC):
    def __init__(self, pos: List[int], paras: float = None):
        super().__init__(pos, paras)

    def get_targ_matrix(self, reverse_order=False):
        targ_matrix = self.matrix

        if reverse_order and (len(self.pos) > 1):
            qnum = len(self.pos)
            dim = 2 ** qnum
            order = np.array(range(len(self.pos))[::-1])
            order = np.concatenate([order, order + qnum])
            tensorm = targ_matrix.reshape([2] * 2 * qnum)
            targ_matrix = np.transpose(tensorm, order).reshape([dim, dim])
        return targ_matrix


class ParaSingleQubitGate(SingleQubitGate, ABC):
    def __init__(self, pos, paras: float):
        if paras is None:
            raise ValueError("`paras` can not be None for ParaSingleQubitGate")
        elif isinstance(paras, int):
            paras = float(paras)
            print(paras)

        if not isinstance(paras, float):
            raise TypeError(f"`paras` must be float or int for ParaSingleQubitGate, instead of {type(paras)}")
        super().__init__(pos, paras=paras)


class FixedSingleQubitGate(SingleQubitGate, ABC):
    def __init__(self, pos):
        super().__init__(pos=pos, paras=None)


class FixedMultiQubitGate(MultiQubitGate, ABC):
    def __init__(self, pos: List[int]):
        super().__init__(pos=pos, paras=None)


class ParaMultiQubitGate(MultiQubitGate, ABC):
    def __init__(self, pos, paras):
        if paras is None:
            raise ValueError("`paras` can not be None for ParaMultiQubitGate")
        super().__init__(pos, paras)


class ControlledGate(MultiQubitGate, ABC):
    """ Controlled gate class, where the matrix act non-trivaly on target qubits"""

    def __init__(self, targe_name, ctrls: List[int], targs: List[int], paras, tar_matrix):
        super().__init__(ctrls + targs, paras)
        self.ctrls = ctrls
        self.targs = targs
        self.targ_name = targe_name
        self._targ_matrix = tar_matrix

        # set matrix
        # TODO: change matrix according to control-type 0/1
        c_n, t_n, n = self.ct_nums
        targ_dim = 2 ** t_n
        dim = 2 ** n
        ctrl_dim = dim - targ_dim
        self._matrix = np.eye(dim, dtype=complex)
        self._matrix[ctrl_dim:, ctrl_dim:] = tar_matrix
        self._matrix = reorder_matrix(self._matrix, self.pos)

        if paras is not None:
            if isinstance(paras, Iterable):
                self.symbol = "%s(" % self.targ_name + ",".join(["%.3f" % para for para in self.paras]) + ")"
            else:
                self.symbol = "%s(%.3f)" % (self.targ_name, paras)
        else:
            self.symbol = "%s" % self.targ_name

    @property
    def matrix(self):
        # TODO: update matrix when paras of controlled-gate changed
        return self._matrix

    @property
    def ct_nums(self):
        targ_num = len(self.targs)
        ctrl_num = len(self.ctrls)
        num = targ_num + ctrl_num
        return ctrl_num, targ_num, num

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
