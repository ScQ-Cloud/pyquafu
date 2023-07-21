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

from quafu.elements.quantum_element.instruction import Instruction

"""
Base classes for ALL kinds of possible instructions on superconducting 
quantum circuits.
"""


class Barrier(Instruction):
    name = "barrier"

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


class Delay(Instruction):
    name = "delay"

    def __init__(self, pos: int, duration: int, unit="ns"):
        if isinstance(duration, int):
            self.duration = duration
        else:
            raise TypeError("duration must be int")
        super().__init__(pos)
        self.unit = unit
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
        super().__init__(list(range(qs, qe + 1)))
        self.unit = unit
        self.symbol = "XY(%d%s)" % (duration, unit)

    def to_qasm(self):
        return "xy(%d%s) " % (self.duration, self.unit) + ",".join(
            ["q[%d]" % p for p in range(min(self.pos), max(self.pos) + 1)])


class Measure(Instruction):
    name = "measure"

    def __init__(self, bitmap: dict):
        super().__init__(list(bitmap.keys()))
        self.qbits = bitmap.keys()
        self.cbits = bitmap.values()


Instruction.register_ins(Barrier)
Instruction.register_ins(Delay)
Instruction.register_ins(XYResonance)
Instruction.register_ins(Measure)
