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

from abc import ABC, abstractmethod
from typing import Union, List


__all__ = ['Instruction', 'Barrier', 'Measure', 'PosType', 'ParaType']

PosType = Union[int, List[int]]
ParaType = Union[float, int, List]


class Instruction(ABC):
    """Base class for ALL the possible instructions on Quafu superconducting quantum circuits.

    Attributes:
        pos: Qubit position(s) of the instruction on the circuit.
        paras: Parameters of the instruction.

    """
    ins_classes = {}

    def __init__(self, pos: PosType, paras: ParaType = None, *args, **kwargs):
        self.pos = pos
        self.paras = paras

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError(
            "name is not implemented for %s" % self.__class__.__name__
            + ", this should never happen."
        )

    @name.setter
    def name(self, _):
        import warnings

        warnings.warn(
            "Invalid assignment, names of standard instructions are not alterable."
        )

    @classmethod
    def register_ins(cls, subclass, name: str = None):
        assert issubclass(subclass, cls)

        if name is None:
            name = subclass.name
        if name in cls.ins_classes:
            raise ValueError(f"Name {name} already exists.")
        cls.ins_classes[name] = subclass


class Barrier(Instruction):
    """
    Barrier instruction.
    """
    name = "barrier"

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

    def __init__(self, pos):
        super().__init__(pos)
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


class Measure(Instruction):
    """
    Measure instruction.
    """
    name = "measure"

    def __init__(self, bitmap: dict):
        super().__init__(list(bitmap.keys()))
        self.qbits = bitmap.keys()
        self.cbits = bitmap.values()


Instruction.register_ins(Barrier)
Instruction.register_ins(Measure)

