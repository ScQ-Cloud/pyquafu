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

# Classes of classical operation.

from typing import Dict, List

from quafu.elements.instruction import Instruction


class Cif(Instruction):
    name = "cif"
    named_paras = None

    def __init__(self, cbits: List[int], condition: int, instructions=None):
        # cbit can be a list of cbit or just a cbit
        self.cbits = cbits
        self.condition = condition
        self.instructions = instructions
        Instruction.__init__(self, pos=-1)

    @property
    def named_pos(self) -> Dict:
        return {"cbits": self.cbits}

    def to_qasm(self, with_para):
        raise NotImplementedError

    def set_ins(self, instructions: List[Instruction]):
        self.instructions = instructions


Instruction.register_ins(Cif)
