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
from typing import Dict, List, Optional, Union 

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
        paras: List[ParameterType] = [],
        *args,
        **kwargs,
    ):
        self.pos = pos
        self.paras = paras
        self._symbol = None

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError(
            "name is not implemented for %s" % self.__class__.__name__
            + ", this should never happen."
        )

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
        import warnings

        warnings.warn(
            "Invalid assignment, names of standard instructions are not alterable."
        )

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
    def named_pos(self):
        return super().named_pos

    @property
    def named_paras(self):
        return super().named_pos

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def to_qasm(self, with_para):
        return "barrier " + ",".join(
            ["q[%d]" % p for p in range(min(self.pos), max(self.pos) + 1)]
        )


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

    def to_qasm(self, with_para):
        return "reset " + ",".join(
            ["q[%d]" % p for p in range(min(self.pos), max(self.pos) + 1)]
        )


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
        lines = [
            "measure q[%d] -> meas[%d];\n" % (q, c)
            for q, c in zip(self.qbits, self.cbits)
        ]
        qasm = "".join(lines)
        return qasm


Instruction.register_ins(Barrier)
Instruction.register_ins(Measure)
