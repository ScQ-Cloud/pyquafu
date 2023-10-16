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
from typing import Union, List, Dict


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
        """Name of the instruction."""
        raise NotImplementedError('name is not implemented for %s' % self.__class__.__name__
                                  + ', this should never happen.')

    @property
    @abstractmethod
    def named_paras(self) -> Dict:
        """dict-mapping for parameters"""
        return {}

    @property
    @abstractmethod
    def named_pos(self) -> Dict:
        """dict-mapping for positions"""
        return {'pos': self.pos}

    @name.setter
    def name(self, _):
        import warnings
        warnings.warn("Invalid assignment, names of standard instructions are not alterable.")

    @classmethod
    def register_ins(cls, subclass, name: str = None):
        assert issubclass(subclass, cls)

        if name is None:
            name = subclass.name
        if name in cls.ins_classes:
            raise ValueError(f"Name {name} already exists.")
        cls.ins_classes[name] = subclass

    @abstractmethod
    def to_qasm(self):
        pass


class Barrier(Instruction):
    """
    Barrier instruction.
    """
    name = "barrier"

    def __init__(self, pos):
        super().__init__(pos)
        self.symbol = "||"

    @property
    def named_pos(self):
        return self.named_pos

    @property
    def named_paras(self):
        return self.named_paras

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
        self.qbits = self.pos
        self.cbits = list(bitmap.values())

    @property
    def named_pos(self):
        return {'pos': self.pos}  # TODO

    @property
    def named_paras(self):
        return self.named_paras

    def to_qasm(self):
        lines = ["measure q[%d] -> meas[%d];\n" % (q, c) for q, c in zip(self.qbits, self.cbits)]
        qasm = ''.join(lines)
        return qasm


Instruction.register_ins(Barrier)
Instruction.register_ins(Measure)
