import quafu.elements.element_gates as qeg
from quafu.elements.quantum_element.quantum_element import (
    Measure,
    Barrier,
    Delay,  
    XYResonance,                                                 
)

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
    def __init__(self, type, node:Node):
        self.type = type
        self.children = [node]

class BinaryExpr(Node):
    def __init__(self, type, left, right):
        self.type = type
        self.children = [left, right]

class Id(Node):
    def __init__(self, name, lineno, filename) :
        self.name = name
        self.lineno = lineno
        self.filename = filename

class IndexedId(Node):
    def __init__(self, node:Node, index):
        self.num = index
        self.name = node.name
        self.lineno = node.lineno
        self.filename = node.filename

class GateInstruction:
    def __init__(self, node, qargs, cargs = None, cbits = None):
        self.name = node.name
        self.lineno = node.lineno
        self.filename = node.filename
        self.qargs = qargs
        self.cargs = cargs
        self.cbits = cbits

class SymtabNode(Node):
    def __init__(self, type, node, is_global=True, is_qubit = False):
        # qreg creg gate arg qarg
        self.type = type
        self.name = node.name
        # for reg
        self.num = node.num
        self.start = 0
        # for error
        self.lineno = node.lineno
        self.filename = node.filename
        # for check
        self.is_global = is_global
        # for gate
        self.is_qubit = is_qubit
        self.qargs = None
        self.cargs = None
        self.instructions = None
    def fill_gate(self, qargs, instructions = None, cargs = None):
        self.qargs = qargs
        self.instructions = instructions
        self.cargs = cargs

        
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
    # "sy": qeg.SYGate,
    # "w": qeg.WGate,
    # "sw": qeg.SWGate,
    "p": qeg.PhaseGate,
    "cx": qeg.CXGate,
    "cnot": qeg.CXGate,
    "cp": qeg.CPGate,
    "swap": qeg.SwapGate,
    "rxx": qeg.RXXGate,
    # "ryy": qeg.RYYGate,
    "rzz": qeg.RZZGate,
    "cy": qeg.CYGate,
    "cz": qeg.CZGate,
    # "cs": qeg.CSGate,
    # "ct": qeg.CTGate,
    "ccx": qeg.ToffoliGate,
    # "toffoli": qeg.ToffoliGate,
    "cswap": qeg.FredkinGate,
    "fredkin": qeg.FredkinGate,
    "mcx": qeg.MCXGate,
    "mcy": qeg.MCYGate,
    "mcz": qeg.MCZGate,
    "delay": Delay,
    "barrier": Barrier,
    "measure" : Measure,
    # "xy": XYResonance,
}