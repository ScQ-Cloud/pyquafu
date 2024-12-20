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
"""Instruction Module."""
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Union

from .parameters import ParameterType

__all__ = ["Instruction", "Barrier", "Measure", "Reset"]


class Instruction(ABC):
    """Base class for ALL the possible instructions on Quafu superconducting quantum circuits.

    Attributes:
        pos: Qubit position(s) of the instruction on the circuit.
        paras: Parameters of the instruction.

    """

    ins_classes = {}

    def __init__(
        self,
        pos: List[int],
        paras: Union[List[ParameterType], None] = None,
        **kwargs,  # pylint: disable=unused-argument
    ):
        if paras is None:
            paras = []
        self.pos = pos
        self.paras = paras
        self._symbol = None

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError(f"name is not implemented for {self.__class__.__name__}, this should never happen.")

    @property
    @abstractmethod
    def named_paras(self) -> Dict:
        """dict-mapping for parameters"""
        return {}

    @property
    @abstractmethod
    def named_pos(self) -> Dict:
        """dict-mapping for positions"""
        return {"pos": self.pos}

    @name.setter
    def name(self, _):
        warnings.warn("Invalid assignment, names of standard instructions are not alterable.")

    @classmethod
    def register_ins(cls, subclass, name: str = None):
        assert issubclass(subclass, cls)

        if name is None:
            name = str(subclass.__name__).lower()
        if name in cls.ins_classes:
            raise ValueError(f"Name {name} already exists.")
        cls.ins_classes[name] = subclass
        return subclass

    @classmethod
    def register(cls, name: str = None):
        def wrapper(subclass):
            return cls.register_ins(subclass, name)

        return wrapper

    @abstractmethod
    def to_qasm(self, with_para):
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
        return super().named_pos

    @property
    def named_paras(self):
        return super().named_pos

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def to_qasm(self, _):
        return "barrier " + ",".join([f"q[{p}]" for p in range(min(self.pos), max(self.pos) + 1)])


class Reset(Instruction):
    name = "reset"

    def __init__(self, pos):
        super().__init__(pos)

    @property
    def pos(self):
        return self.__pos

    @pos.setter
    def pos(self, pos):
        self.__pos = pos

    @property
    def named_pos(self):
        return self.named_pos

    @property
    def named_paras(self):
        return self.named_paras

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def to_qasm(self, _):
        return "reset " + ",".join([f"q[{p}]" for p in range(min(self.pos), max(self.pos) + 1)])


class Measure(Instruction):
    """
    Measure instruction.
    """

    name = "measure"

    def __init__(self, bitmap: dict):
        super().__init__(list(bitmap.keys()))
        self.qbits = list(bitmap.keys())
        self.cbits = list(bitmap.values())

    @property
    def named_pos(self):
        return {"pos": self.pos}  # TODO

    @property
    def named_paras(self):
        return self.named_paras

    def to_qasm(self, with_para):
        lines = [f"measure q[{q}] -> meas[{c}];\n" for q, c in zip(self.qbits, self.cbits)]
        return "".join(lines)


Instruction.register_ins(Barrier)
Instruction.register_ins(Measure)
