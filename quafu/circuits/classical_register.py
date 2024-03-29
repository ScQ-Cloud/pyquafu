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


class ClassicalRegister:
    """
    Collection of cbit(s)
    """

    def __init__(self, num: int = 0, name: str = None):
        self.name = name
        self.num = num
        self.cbits = {i: 0 for i in range(num)}
        self.pos_start = 0

    def __getitem__(self, item):
        """Get mapped global pos"""
        if item < self.num:
            return self.pos_start + item
        else:
            raise IndexError("Index out of range:", item)

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i < self.num:
            x = self._i
            self._i += 1
            return self.__getitem__(x)
        else:
            raise StopIteration

    def __len__(self):
        return self.num

    @property
    def value(self, item: int = None):
        """Get value stored in register"""
        if item is None:
            return self.cbits
        if item >= self.num or item < 0:
            raise Exception(f"index {item} out of range.")
        return self.cbits[item]

    def __add__(self, other: "ClassicalRegister"):
        creg = ClassicalRegister(name=self.name)
        creg.cbits = {
            **{self.cbits},
            **{i + len(self): cbit for i, cbit in other.cbits.items()},
        }
        creg.num = self.num + other.num
        creg.pos_start = self.pos_start
        return creg
