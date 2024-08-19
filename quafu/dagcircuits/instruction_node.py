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

from typing import Dict, Any, List, Union
import dataclasses


@dataclasses.dataclass
class InstructionNode:
    """
    A class representing a single instruction in a quantum circuit.

    Attributes:
    -----------
    name : Any
        The name of the gate.
    pos : Union[List[Any], Dict[Any, Any]]
        The position of the gate in the circuit. If the gate is a measurement gate, it is a dictionary with qubit indices as keys and classical bit indices as values.
    paras : List[Any]
        The parameters of the gate.
    duration : int
        The duration of the gate. Only applicable for certain gates.
    unit : str
        The unit of the duration. Only applicable for certain gates.
    label : Union[str, int]
        The label of the instruction node.

    Methods:
    --------
    __hash__(self)
        Returns the hash value of the instruction node.
    __str__(self)
        Returns a string representation of the instruction node.
    __repr__(self)
        Returns a string representation of the instruction node.
    """
    name: Any  # gate.name
    pos: Union[List[Any], Dict[Any, Any]]  # gate.pos |  Dict[Any,Any] for measure
    paras: List[Any]  # gate.paras
    # matrix:List[Any]   # for gate in [QuantumGate]
    duration: int  # for gate in [Delay,XYResonance] in quafu
    unit: str  # for gate in [Delay,XYResonance] in quafu
    label: Union[str, int]

    def __hash__(self):
        return hash((type(self.name), tuple(self.pos), self.label))

    def __str__(self):
        if self.name == 'measure':
            args = ','.join(str(q) for q in self.pos.keys())
            args += f'=>{",".join(str(c) for c in self.pos.values())}'
        else:
            args = ','.join(str(q) for q in self.pos)

        if self.paras is None:
            return f'{self.label}{{{self.name}({args})}}'
        else:

            # if self.paras not a list, then make it a list  of str of .3f float
            if not isinstance(self.paras, list):
                if isinstance(self.paras, float):
                    formatted_paras = [f'{self.paras:.3f}']
                else:
                    formatted_paras = [str(self.paras)]
            else:
                if all(isinstance(p, float) for p in self.paras):
                    formatted_paras = [f'{p:.3f}' for p in self.paras]
                else:
                    # for p in self.paras:
                    #        print(p, type(p))
                    formatted_paras = [str(p) for p in self.paras]
                    # print(formatted_paras)

            formatted_paras_str = ','.join(formatted_paras)

            return f'{self.label}{{{self.name}({args})}}({formatted_paras_str})'

    def __repr__(self):
        return str(self)
