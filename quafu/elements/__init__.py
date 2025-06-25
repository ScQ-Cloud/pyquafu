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
"""Quantum circuit elements module."""

from .classical_element import Cif
from .instruction import Barrier, Instruction, Measure, Reset
from .noise import KrausChannel, UnitaryChannel
from .oracle import ControlledOracle, OracleGate
from .parameters import Parameter, ParameterExpression, ParameterType
from .pulses import Delay, QuantumPulse, XYResonance
from .quantum_gate import (
    CircuitWrapper,
    ControlledCircuitWrapper,
    ControlledGate,
    QuantumGate,
)
from .unitary import UnitaryDecomposer
from .utils import extract_float, reorder_matrix

__all__ = [
    "Cif",
    "Barrier",
    "CircuitWrapper",
    "ControlledCircuitWrapper",
    "ControlledGate",
    "ControlledOracle",
    "Instruction",
    "KrausChannel",
    "Measure",
    "OracleGate",
    "Parameter",
    "ParameterExpression",
    "ParameterType",
    "QuantumGate",
    "QuantumPulse",
    "Reset",
    "UnitaryChannel",
    "UnitaryDecomposer",
    "XYResonance",
    "extract_float",
    "reorder_matrix",
    "Delay",
]
