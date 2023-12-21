from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Union

import numpy as np
from quafu.elements.matrices.mat_utils import reorder_matrix

from .instruction import Instruction, PosType


class QuantumGate(Instruction, ABC):
    """Base class for standard and combined quantum gates.

    Attributes:

    """

    gate_classes = {}

    def __init__(
        self,
        pos: PosType,
        paras: Union[float, List[float]] = None,
    ):
        super().__init__(pos, paras)

        if paras is not None:
            if isinstance(paras, Iterable):
                self.symbol = (
                    "%s(" % self.name
                    + ",".join(["%.3f" % para for para in self.paras])
                    + ")"
                )
            else:
                self.symbol = "%s(%.3f)" % (self.name, paras)
        else:
            self.symbol = "%s" % self.name

    def update_params(self, paras: Union[float, List[float]]):
        """Update parameters of this gate"""
        if paras is None:
            return
        self.paras = paras
        if isinstance(paras, Iterable):
            self.symbol = (
                "%s(" % self.name
                + ",".join(["%.3f" % para for para in self.paras])
                + ")"
            )
        else:
            self.symbol = "%s(%.3f)" % (self.name, paras)

    @property
    @abstractmethod
    def matrix(self):
        raise NotImplementedError(
            "Matrix is not implemented for %s" % self.__class__.__name__
            + ", this should never happen."
        )

    @classmethod
    def register_gate(cls, subclass, name: str = None):
        assert issubclass(subclass, cls)

        name = str(subclass.name).lower() if name is None else name
        assert isinstance(name, str)

        if name in cls.gate_classes:
            raise ValueError(f"Name {name} already exists.")
        cls.gate_classes[name] = subclass
        Instruction.register_ins(subclass, name)

    @classmethod
    def register(cls, name: str = None):
        def wrapper(subclass):
            cls.register_gate(subclass, name)
            return subclass

        return wrapper

    def __str__(self):
        properties_names = ["pos", "paras", "matrix"]
        properties_values = [getattr(self, x) for x in properties_names]
        return "%s:\n%s" % (
            self.__class__.__name__,
            "\n".join(
                [
                    f"{x} = {repr(properties_values[i])}"
                    for i, x in enumerate(properties_names)
                ]
            ),
        )

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def to_qasm(self):
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


# Gate types are statically implemented to support type identification
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
        return {"pos": self.pos}


class MultiQubitGate(QuantumGate, ABC):
    def __init__(self, pos: List, paras: float = None):
        QuantumGate.__init__(self, pos, paras)

    def get_targ_matrix(self, reverse_order=False):
        """ """
        targ_matrix = self.matrix

        if reverse_order and (len(self.pos) > 1):
            qnum = len(self.pos)
            dim = 2**qnum
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
        return {"paras": self.paras}

    @property
    def named_pos(self) -> Dict:
        return {"pos": self.pos}


class ControlledGate(MultiQubitGate, ABC):
    """Controlled gate class, where the matrix act non-trivially on target qubits"""

    def __init__(
        self, targe_name, ctrls: List[int], targs: List[int], paras, tar_matrix
    ):
        MultiQubitGate.__init__(self, ctrls + targs, paras)
        self.ctrls = ctrls
        self.targs = targs
        self.targ_name = targe_name
        self._targ_matrix = tar_matrix

        # set matrix
        # TODO: change matrix according to control-type 0/1
        c_n, t_n, n = self.ct_nums
        targ_dim = 2**t_n
        dim = 2**n
        ctrl_dim = dim - targ_dim
        self._matrix = np.eye(dim, dtype=complex)
        self._matrix[ctrl_dim:, ctrl_dim:] = tar_matrix
        self._matrix = reorder_matrix(self._matrix, self.pos)

        if paras is not None:
            if isinstance(paras, Iterable):
                self.symbol = (
                    "%s(" % self.targ_name
                    + ",".join(["%.3f" % para for para in self.paras])
                    + ")"
                )
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
            dim = 2**qnum
            tensorm = targ_matrix.reshape([2] * 2 * qnum)
            targ_matrix = np.transpose(tensorm, order).reshape([dim, dim])
        return targ_matrix

    @property
    def named_pos(self) -> Dict:
        return {"ctrls": self.ctrls, "targs": self.targs}


# class ParaSingleQubitGate(SingleQubitGate, ABC):
#     def __init__(self, pos, paras: float):
#         if paras is None:
#             raise ValueError("`paras` can not be None for ParaSingleQubitGate")
#         elif isinstance(paras, int):
#             paras = float(paras)
#
#         if not isinstance(paras, float):
#             raise TypeError(f"`paras` must be float or int for ParaSingleQubitGate, "
#                             f"instead of {type(paras)}")
#         super().__init__(pos, paras=paras)
#
#     @property
#     def named_paras(self) -> Dict:
#         return {'paras': self.paras}

# class FixedMultiQubitGate(MultiQubitGate, ABC):
#     def __init__(self, pos: List[int]):
#         super().__init__(pos=pos, paras=None)

# class ParaMultiQubitGate(MultiQubitGate, ABC):
#     def __init__(self, pos, paras):
#         if paras is None:
#             raise ValueError("`paras` can not be None for ParaMultiQubitGate")
#         super().__init__(pos, paras)


class FixedGate(QuantumGate, ABC):
    def __init__(self, pos):
        super().__init__(pos=pos, paras=None)

    @property
    def named_paras(self) -> Dict:
        return {}
