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

from quafu.circuits.quantum_circuit import QuantumCircuit

from .qfasm_parser import QfasmParser


def qasm_to_quafu(openqasm: str):
    """
    Initialize pyquafu circuit from openqasm text, mainly by
    utilizing regular expressions. This function is developed
    not only to be backward compatible with pure quantum part of
    OPENQASM2.0, but also to support provide foundation of more
    powerful syntax parser.
    """
    qparser = QfasmParser()
    qc = qparser.parse(openqasm)
    qc.openqasm = openqasm
    if not qc.executable_on_backend:
        print(
            "Warning: At present, quafu's back-end real machine does not support dynamic circuits."
        )
    return qc


def qasm2_to_quafu_qc(qc: QuantumCircuit, openqasm: str):
    """
    Initialize pyquafu circuit from openqasm text, mainly by
    utilizing regular expressions. This function is developed
    not only to be backward compatible with pure quantum part of
    OPENQASM2.0, but also to support provide foundation of more
    powerful syntax parser.
    """
    qparser = QfasmParser()
    newqc = qparser.parse(openqasm)
    qc.openqasm = openqasm
    qc.gates = newqc.gates
    qc.instructions = newqc.instructions
    qc._measures = newqc._measures
    qc.qregs = newqc.qregs
    qc.cregs = newqc.cregs
    qc.executable_on_backend = newqc.executable_on_backend
    if not qc.executable_on_backend:
        print(
            "Warning: At present, quafu's back-end real machine does not support dynamic circuits."
        )
