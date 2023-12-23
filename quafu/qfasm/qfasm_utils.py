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

import quafu.elements.element_gates as qeg


class Node:
    """
    Node created in reduction.
    Help for building QuantumCrcuit.
    """

    def __init__(self):
        self.type = None
        self.name = None
        self.num = 0
        self.start = 0
        self.lineno = 0
        self.filename = None


class UnaryExpr(Node):
    def __init__(self, type, node: Node):
        self.type = type
        self.children = [node]


class BinaryExpr(Node):
    def __init__(self, type, left, right):
        self.type = type
        self.children = [left, right]


class Id(Node):
    def __init__(self, name, lineno, filename):
        self.name = name
        self.lineno = lineno
        self.filename = filename
        self.num = 0


class IndexedId(Node):
    def __init__(self, node: Node, index):
        self.num = index
        self.name = node.name
        self.lineno = node.lineno
        self.filename = node.filename


class GateInstruction:
    def __init__(self, node, qargs, cargs=None, cbits=None):
        self.name = node.name
        self.lineno = node.lineno
        self.filename = node.filename
        self.qargs = qargs
        self.cargs = cargs
        self.cbits = cbits


class IfInstruction:
    def __init__(self, node, cbits, value: int, instruction):
        self.name = node.name
        self.lineno = node.lineno
        self.filename = node.filename
        self.cbits = cbits
        self.value = value
        self.instruction = instruction


class SymtabNode(Node):
    def __init__(self, type, node, is_global=True, is_qubit=False):
        # qreg creg gate arg qarg
        self.type = type
        self.name = node.name
        # for reg
        self.num = 0
        if isinstance(node, IndexedId):
            self.num = node.num
        self.start = 0
        # for debug
        self.lineno = node.lineno
        self.filename = node.filename
        # for check
        self.is_global = is_global
        # for gate
        self.is_qubit = is_qubit
        self.qargs = []
        self.cargs = []
        self.instructions = []

    def fill_gate(self, qargs, instructions=None, cargs=None):
        self.qargs = qargs
        self.instructions = instructions
        if cargs:
            self.cargs = cargs


from quafu.elements import Barrier, Delay, Measure, XYResonance

gate_classes = {
    "x": qeg.XGate,
    "y": qeg.YGate,
    "z": qeg.ZGate,
    "h": qeg.HGate,
    "s": qeg.SGate,
    "sdg": qeg.SdgGate,
    "t": qeg.TGate,
    "tdg": qeg.TdgGate,
    "rx": qeg.RXGate,
    "ry": qeg.RYGate,
    "rz": qeg.RZGate,
    "id": qeg.IdGate,
    "sx": qeg.SXGate,
    "sy": qeg.SYGate,
    "w": qeg.WGate,
    "sw": qeg.SWGate,
    "p": qeg.PhaseGate,
    "cx": qeg.CXGate,
    "cnot": qeg.CXGate,
    "cp": qeg.CPGate,
    "swap": qeg.SwapGate,
    "rxx": qeg.RXXGate,
    "ryy": qeg.RYYGate,
    "rzz": qeg.RZZGate,
    "cy": qeg.CYGate,
    "cz": qeg.CZGate,
    "cs": qeg.CSGate,
    "ct": qeg.CTGate,
    "ccx": qeg.ToffoliGate,
    "toffoli": qeg.ToffoliGate,
    "cswap": qeg.FredkinGate,
    "fredkin": qeg.FredkinGate,
    "mcx": qeg.MCXGate,
    "mcy": qeg.MCYGate,
    "mcz": qeg.MCZGate,
    "delay": Delay,
    "barrier": Barrier,
    "measure": Measure,
    "xy": XYResonance,
}
