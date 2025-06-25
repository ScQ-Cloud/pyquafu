# (C) Copyright 2024 Beijing Academy of Quantum Information Sciences
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
"""Noise channel module."""

import copy
from typing import List, Union

import numpy as np

from .element_gates import IdGate, XGate, YGate, ZGate
from .instruction import Instruction
from .quantum_gate import QuantumGate


class KrausChannel(Instruction):
    def __init__(self, name, pos: int, gatelist: Union[None, List[QuantumGate]] = None):
        if gatelist is None:
            gatelist = []
        self._name = name
        self._pos = [pos]
        self.gatelist = gatelist

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, __name):
        self._name = __name

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, __pos):
        self._pos = copy.deepcopy(__pos)

    @property
    def named_pos(self):
        return {"pos": self.pos}

    @property
    def named_paras(self):
        return {"paras": self.paras}

    def to_qasm(self, with_para):
        raise ValueError("Can not convert noise channel to qasm")

    @property
    def symbol(self):
        if len(self.paras) > 0:
            return f"{self.name}({','.join([f'{para:.3f}' for para in self.paras])})"
        return f"{self.name}"

    def __repr__(self):
        return self.symbol


class UnitaryChannel(KrausChannel):
    def __init__(
        self,
        name,
        pos: int,
        gatelist: Union[None, List[QuantumGate]] = None,
        probs: Union[None, List[float]] = None,
    ):
        if gatelist is None:
            gatelist = []
        if probs is None:
            probs = []
        super().__init__(name, pos, gatelist)
        self.probs = probs

    def gen_gate(self):
        """
        Randomly choose a gate
        """
        return np.random.choice(self.gatelist, p=self.probs)


@Instruction.register()
class BitFlip(UnitaryChannel):
    name = "BitFlip"

    def __init__(self, pos: int, p):
        self.pos = [pos]
        self.paras = [p]
        self.gatelist = [XGate(pos), IdGate(pos)]
        self.probs = [p, 1 - p]


@Instruction.register()
class Dephasing(UnitaryChannel):
    name = "Dephasing"

    def __init__(self, pos: int, p):
        self.pos = [pos]
        self.paras = [p]
        self.gatelist = [ZGate(pos), IdGate(pos)]
        self.probs = [p, 1 - p]


@Instruction.register()
class Depolarizing(UnitaryChannel):
    name = "Depolarizing"

    def __init__(self, pos: int, p):
        self.pos = [pos]
        self.paras = [p]
        self.gatelist = [XGate(pos), YGate(pos), ZGate(pos), IdGate(pos)]
        self.probs = [p / 3, p / 3, p / 3, 1 - p]


@Instruction.register()
class AmplitudeDamping(KrausChannel):
    name = "AmpDamping"

    def __init__(self, pos: int, p):
        self.pos = [pos]
        self.paras = [p]
        dampmat0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - p)]], dtype=complex)
        dampgate0 = QuantumGate("AMPDAMP0", [pos], [], dampmat0)
        dampmat1 = np.array([[0.0, np.sqrt(p)], [0.0, 0.0]], dtype=complex)
        dampgate1 = QuantumGate("AMPDAMP1", [pos], [], dampmat1)
        self.gatelist = [dampgate0, dampgate1]


@Instruction.register()
class Decoherence(KrausChannel):
    name = "Decoherence"

    def __init__(self, pos, t, T1, T2):
        """
        t: time of quantum operation
        T1: energy decay time
        T2: dephasing time
        """
        self.pos = [pos]
        self.paras = [t, T1, T2]
        kmat0 = np.array([[1.0, 0.0], [0.0, np.exp(-t / T2)]], dtype=complex)
        kmat1 = np.array(
            [[0.0, np.sqrt(1 - np.exp(-t / T1))], [0.0, 0.0]], dtype=complex
        )
        kmat2 = np.array(
            [[1.0, 0.0], [0.0, np.sqrt(np.exp(-t / T1) - np.exp(-2 * t / T2))]],
            dtype=complex,
        )
        kgate0 = QuantumGate("DECAY0", [pos], [], kmat0)
        kgate1 = QuantumGate("DECAY1", [pos], [], kmat1)
        kgate2 = QuantumGate("DECAY2", [pos], [], kmat2)
        self.gatelist = [kgate0, kgate1, kgate2]
