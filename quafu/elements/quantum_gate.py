from abc import ABC, abstractmethod
from typing import List, Union, Iterable, Dict, Callable

import numpy as np

from quafu.elements.matrices.mat_utils import reorder_matrix
from .instruction import Instruction, PosType


__all__ = ['QuantumGate', 'FixedGate', 'ParametricGate', 'SingleQubitGate', 'MultiQubitGate', 'ControlledGate']


class QuantumGate(Instruction, ABC):
    """Base class for standard and combined quantum gates, namely unitary operation
    upon quantum states.

    Attributes:
        pos: Position of this gate in the circuit.
        paras: Parameters of this gate.

    Properties:
        matrix: Matrix representation of this gate.
    """
    gate_classes = {}

    def __init__(self,
                 pos: PosType,
                 paras: Union[float, List[float]] = None,
                 matrix: Union[np.ndarray, Callable] = None,
                 ):
        super().__init__(pos, paras)
        self._matrix = matrix

    @property
    def symbol(self):
        if self.paras is not None:
            if isinstance(self.paras, Iterable):
                symbol = "%s(" % self.name + ",".join(["%.3f" % para for para in self.paras]) + ")"
            else:
                symbol = "%s(%.3f)" % (self.name, self.paras)
        else:
            symbol = "%s" % self.name
        return symbol

    def update_params(self, paras: Union[float, List[float]]):
        """Update parameters of this gate"""
        if paras is None:
            return
        self.paras = paras

    @property
    @abstractmethod
    def matrix(self):
        if self._matrix is not None:
            return self._matrix
        else:
            raise NotImplementedError("Matrix is not implemented for %s" % self.__class__.__name__ +
                                      ", this should never happen.")

    @classmethod
    def register_gate(cls, subclass, name: str = None):
        """Register a new gate class into gate_classes.

        This method is used as a decorator.
        """
        assert issubclass(subclass, cls)

        name = str(subclass.name).lower() if name is None else name
        assert isinstance(name, str)

        if name in cls.gate_classes:
            raise ValueError(f"Name {name} already exists.")
        cls.gate_classes[name] = subclass
        Instruction.register_ins(subclass, name)

    @classmethod
    def register(cls, name: str = None):
        """Decorator for register_gate."""
        def wrapper(subclass):
            cls.register_gate(subclass, name)
            return subclass

        return wrapper

    def __str__(self):
        # only when the gate is a known(named) gate, the matrix is not shown
        if self.name.lower() in self.gate_classes:
            properties_names = ['pos', 'paras']
        else:
            properties_names = ['pos', 'paras', 'matrix']
        properties_values = [getattr(self, x) for x in properties_names]
        return "%s:\n%s" % (self.__class__.__name__, '\n'.join(
            [f"{x} = {repr(properties_values[i])}" for i, x in enumerate(properties_names)]))

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def to_qasm(self):
        # TODO: support register naming
        qstr = "%s" % self.name.lower()

        if self.paras is not None:
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


# Gate types below are statically implemented to support type identification
# and provide shared attributes. However, single/multi qubit may be
# inferred from ``pos``, while para/fixed type may be inferred by ``paras``.
# Therefore, these types may be (partly) deprecated in the future.

class SingleQubitGate(QuantumGate, ABC):
    def __init__(self, pos: int, paras: float = None):
        QuantumGate.__init__(self, pos=pos, paras=paras)

    def get_targ_matrix(self):
        return self.matrix

    @property
    def named_pos(self) -> Dict:
        return {'pos': self.pos}


class MultiQubitGate(QuantumGate, ABC):
    def __init__(self, pos: List, paras: float = None):
        QuantumGate.__init__(self, pos, paras)

    def get_targ_matrix(self, reverse_order=False):
        """

        """
        targ_matrix = self.matrix

        if reverse_order and (len(self.pos) > 1):
            qnum = len(self.pos)
            dim = 2 ** qnum
            order = np.array(range(len(self.pos))[::-1])
            order = np.concatenate([order, order + qnum])
            tensorm = targ_matrix.reshape([2] * 2 * qnum)
            targ_matrix = np.transpose(tensorm, order).reshape([dim, dim])
        return targ_matrix


class ParametricGate(QuantumGate, ABC):
    def __init__(self, pos: PosType, paras: Union[float, List[float]]):
        if paras is None:
            raise ValueError("`paras` can not be None for ParametricGate")
        super().__init__(pos, paras)

    @property
    def named_paras(self) -> Dict:
        return {'paras': self.paras}

    @property
    def named_pos(self) -> Dict:
        return {'pos': self.pos}


class FixedGate(QuantumGate, ABC):
    def __init__(self, pos):
        super().__init__(pos=pos, paras=None)

    @property
    def named_paras(self) -> Dict:
        return {}


class ControlledGate(MultiQubitGate, ABC):
    """ Controlled gate class, where the matrix act non-trivially on target qubits"""

    def __init__(self, targe_name, ctrls: List[int], targs: List[int], paras, tar_matrix):
        MultiQubitGate.__init__(self, ctrls + targs, paras)
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
        self._matrix[ctrl_dim:, ctrl_dim:] = self.targ_matrix
        self._matrix = reorder_matrix(self._matrix, self.pos)

    @property
    def symbol(self):
        name = self.targ_name
        if self.paras is not None:
            if isinstance(self.paras, Iterable):
                symbol = "%s(" % name + ",".join(["%.3f" % para for para in self.paras]) + ")"
            else:
                symbol = "%s(%.3f)" % (name, self.paras)
        else:
            symbol = "%s" % name
        return symbol

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

    @property
    def targ_matrix(self):
        if isinstance(self._targ_matrix, Callable):
            return self._targ_matrix(self.paras)
        else:
            return self._targ_matrix

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

    @property
    def named_pos(self) -> Dict:
        return {'ctrls': self.ctrls, 'targs': self.targs}
