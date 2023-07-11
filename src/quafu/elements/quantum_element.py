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

from abc import ABC
from typing import Union, Callable, List, Iterable

import numpy as np

"""
Base classes for ALL kinds of possible instructions on superconducting 
quantum circuits.
"""

ParaType = dict[str: float | int]
PosType = Union[int, List[int]]


def reorder_matrix(matrix: np.ndarray, pos: List):
    """Reorder the input sorted matrix to the pos order """
    qnum = len(pos)
    dim = 2 ** qnum
    inds = np.argsort(pos)
    inds = np.concatenate([inds, inds + qnum])
    tensorm = matrix.reshape([2] * 2 * qnum)
    return np.transpose(tensorm, inds).reshape([dim, dim])


class Instruction(ABC):
    name = None  # type: str
    # """
    # Metaclass for primitive instructions in DIRECTED ACYCLIC quantum circuits
    # or quantum-classical circuits.
    #
    # Member variables:
    #
    # - sd_name: standard name without args or extra annotations. Used for communications
    #   and identifications within pyquafu program, for example, to translate a qu_gate into qasm.
    #   Ideally every known primitive during computation should have such a name, and should be
    #   chosen as the most commonly accepted convention. NOT allowed to be changed by users once
    #   the class is instantiated.
    #
    # - pos: positions of relevant quantum or classical bits, legs of DAG.
    #
    # - dorder: depth order, or topological order of DAG
    #
    # - symbol: label that can be freely customized by users. If sd_name is not None, name is the
    #   same as sd_name by default. Otherwise, a symbol has to be specified while sd_name remains
    #   as None to indicate that this is a use-defined class.
    # """
    #
    # _ins_id = None  # type: str
    #
    # def __init__(self, pos: PosType, label: str = None, paras: ParaType = None):
    #     # if pos is not iterable, make it be
    #     self.pos = [pos] if isinstance(pos, int) else pos
    #     if label:
    #         self.label = label
    #     else:
    #         if self._ins_id is None:
    #             raise ValueError('For user-defined instruction, label has to be specified.')
    #         self.label = self._ins_id
    #         if paras:
    #             self.label += '(' + ', '.join(['%.3f' % _ for _ in paras.values()]) + ')'
    #     self.paras = paras
    #
    # @classmethod
    # def get_ins_id(cls):
    #     return cls._ins_id
    #
    # @abstractmethod
    # def openqasm2(self) -> str:
    #     pass

    # @_ins_id.setter
    # def sd_name(self, name: str):
    #     if self.sd_name is None:
    #         self.sd_name = name
    #     else:
    #         import warnings
    #         warnings.warn(message='Invalid assignment, names of standard '
    #                               'instructions are not alterable.')

    # def to_dag_node(self):
    #     name = self.get_ins_id()
    #     label = self.__repr__()
    #
    #     pos = self.pos
    #     paras = self.paras
    #     paras = {} if paras is None else paras
    #     duration = paras.get('duration', None)
    #     unit = paras.get('unit', None)
    #     channel = paras.get('channel', None)
    #     time_func = paras.get('time_func', None)
    #
    #     return InstructionNode(name, pos, paras, duration, unit, channel, time_func, label)


class Barrier(Instruction):
    name = "barrier"

    def __init__(self, pos):
        self.__pos = pos
        self.symbol = "||"

    @property
    def pos(self):
        return self.__pos

    @pos.setter
    def pos(self, pos):
        self.__pos = pos

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def to_qasm(self):
        return "barrier " + ",".join(["q[%d]" % p for p in range(min(self.pos), max(self.pos) + 1)])


class Delay(Instruction):
    name = "delay"

    def __init__(self, pos: int, duration: int, unit="ns"):
        if isinstance(duration, int):
            self.duration = duration
        else:
            raise TypeError("duration must be int")
        self.unit = unit
        self.pos = pos
        self.symbol = "Delay(%d%s)" % (duration, unit)

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def to_qasm(self):
        return "delay(%d%s) q[%d]" % (self.duration, self.unit, self.pos)


class XYResonance(Instruction):
    name = "XY"

    def __init__(self, qs: int, qe: int, duration: int, unit="ns"):
        if isinstance(duration, int):
            self.duration = duration
        else:
            raise TypeError("duration must be int")
        self.unit = unit
        self.pos = list(range(qs, qe + 1))
        self.symbol = "XY(%d%s)" % (duration, unit)

    def to_qasm(self):
        return "xy(%d%s) " % (self.duration, self.unit) + ",".join(
            ["q[%d]" % p for p in range(min(self.pos), max(self.pos) + 1)])


class Measure(Instruction):
    name = "measure"

    def __init__(self, bitmap: dict):
        self.qbits = bitmap.keys()
        self.cbits = bitmap.values()


class QuantumGate(Instruction, ABC):
    name = 'gate'

    def __init__(self,
                 pos: PosType,
                 paras: Union[float, List[float]] = None,
                 matrix=None):
        self.pos = pos
        self.paras = paras
        self.matrix = matrix

        if paras:
            if isinstance(paras, Iterable):
                self.symbol = "%s(" % self.name + ",".join(["%.3f" % para for para in self.paras]) + ")"
            else:
                self.symbol = "%s(%.3f)" % (self.name, paras)
        else:
            self.symbol = "%s" % self.name

    @property
    def pos(self):
        return self.__pos

    @pos.setter
    def pos(self, _pos):
        self.__pos = _pos

    @property
    def paras(self):
        return self.__paras

    @paras.setter
    def paras(self, _paras):
        self.__paras = _paras

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        self._matrix = matrix

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


class SingleQubitGate(QuantumGate):
    name = 'su(2)'

    def __init__(self, pos: int, paras, matrix):
        super().__init__(pos, paras=paras, matrix=matrix)

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        if isinstance(matrix, (np.ndarray, List)):
            if np.shape(matrix) == (2, 2):
                self._matrix = np.asarray(matrix, dtype=complex)
            else:
                raise ValueError(f'`{self.__class__.__name__}.matrix.shape` must be (2, 2)')
        elif isinstance(matrix, type(None)):
            self._matrix = matrix
        else:
            raise TypeError("Unsupported `matrix` type")

    def get_targ_matrix(self, reverse_order=False):
        return self.matrix


class FixedSingleQubitGate(SingleQubitGate):
    def __init__(self, pos, matrix):
        super().__init__(pos, paras=None, matrix=matrix)


class ParaSingleQubitGate(SingleQubitGate):
    def __init__(self, pos, paras, matrix):
        super().__init__(pos, paras=paras, matrix=matrix)

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        if isinstance(matrix, Callable):
            self._matrix = matrix(self.paras)
        elif isinstance(matrix, (np.ndarray, List)):
            if np.shape(matrix) == (2, 2):
                self._matrix = np.asarray(matrix, dtype=complex)
            else:
                raise ValueError(f'`{self.__class__.__name__}.matrix.shape` must be (2, 2)')
        elif isinstance(matrix, type(None)):
            self._matrix = matrix
        else:
            raise TypeError("Unsupported `matrix` type")


class MultiQubitGate(QuantumGate):
    name = 'su(n)'

    def __init__(self, pos: List[int], paras, matrix):
        super().__init__(pos, paras=paras, matrix=matrix)
        self._targ_matrix = matrix

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        if isinstance(matrix, np.ndarray):
            self._matrix = matrix
        elif isinstance(matrix, List):
            self._matrix = np.array(matrix, dtype=complex)
        else:
            raise TypeError("Unsupported `matrix` type")

        self._matrix = reorder_matrix(self._matrix, self.pos)

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


class FixedMultiQubitGate(MultiQubitGate):
    def __init__(self, pos: List[int], matrix):
        super().__init__(pos, paras=None, matrix=matrix)


class ParaMultiQubitGate(MultiQubitGate):
    def __init__(self, pos, paras, matrix):
        super().__init__(pos, paras, matrix=matrix)

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


class ControlledGate(MultiQubitGate):
    """ Controlled gate class, where the matrix act non-trivaly on target qubits"""

    def __init__(self, targe_name, ctrls: List[int], targs: List[int], paras, matrix):
        self.ctrls = ctrls
        self.targs = targs
        self.targ_name = targe_name
        super().__init__(ctrls + targs, paras, matrix)
        self._targ_matrix = matrix

        if paras:
            if isinstance(paras, Iterable):
                self.symbol = "%s(" % self.targ_name + ",".join(["%.3f" % para for para in self.paras]) + ")"
            else:
                self.symbol = "%s(%.3f)" % (self.targ_name, paras)
        else:
            self.symbol = "%s" % self.targ_name

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


class ControlledU(ControlledGate):
    """ Controlled gate class, where the matrix act non-trivaly on target qubits"""

    name = 'cu'

    def __init__(self, ctrls: List[int], U: Union[SingleQubitGate, MultiQubitGate]):
        self.targ_gate = U
        targs = U.pos
        if isinstance(targs, int):
            targs = [targs]

        super().__init__(U.name, ctrls, targs, U.paras, matrix=self.targ_gate.get_targ_matrix())

    def get_targ_matrix(self, reverse_order=False):
        return self.targ_gate.get_targ_matrix(reverse_order)
