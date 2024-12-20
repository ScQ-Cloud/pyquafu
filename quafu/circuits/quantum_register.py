# (C) Copyright 2023 Beijing Academy of Quantum Information Sciences
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict


class Qubit:
    """
    Representation of logical qubits.
    """

    def __init__(
        self,
        logic_pos: int,
        reg_name: str = None,
        label: str = None,
    ):
        self.pos = logic_pos
        self.reg_name = "q" if reg_name is None else reg_name
        self.label = self.reg_name if label is None else label
        self.physical_info = None
        self._depth = 0  # present depth

    def __repr__(self):
        return self.reg_name + "_%s" % self.pos

    def load_physical_info(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def used(self):
        return self._depth > 0

    def add_depth(self, num: int = 1):
        self._depth += num

    def move_pos(self, new_pos):
        old_pos = 0 + self.pos
        self.pos = new_pos
        return old_pos


class QuantumRegister:
    """
    Collection of Qubit(s)
    """

    def __init__(self, num: int = 0, name: str = None):
        self.name = name
        self.qubits = OrderedDict(
            {i: Qubit(logic_pos=i, reg_name=name) for i in range(num)}
        )

    def __getitem__(self, item):
        if item < len(self.qubits):
            return self.qubits[item]
        else:
            raise IndexError("Index out of range:", item)

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i < len(self):
            x = self._i
            self._i += 1
            return self.qubits[x]
        else:
            raise StopIteration

    def __len__(self):
        return len(self.qubits)

    def __add__(self, other: "QuantumRegister"):
        qreg = QuantumRegister(name=self.name)
        qreg.qubits = {
            **{self.qubits},
            **{i + len(self): qubit for i, qubit in other.qubits.items()},
        }
        return QuantumRegister(len(self) + len(other), name=self.name)

    def exchange(self, p1, p2):
        pass
