import ply.yacc as yacc

from quafu.elements.element_gates import *
from quafu.elements.quantum_element import *

from qfasmlex import QfasmLexer
tokens = QfasmLexer.tokens

instructions_map = {
    "x": XGate,
    "y": YGate,
    "z": ZGate,
    "h": HGate,
    "s": SGate,
    "sdg": SdgGate,
    "t": TGate,
    "tdg": TdgGate,
    "rx": RXGate,
    "ry": RYGate,
    "rz": RZGate,
    "id": IdGate,
    "sx": SXGate,
    "sy": SYGate,
    "w": WGate,
    "sw": SWGate,
    "p": PhaseGate,
    "delay": Delay,
    "barrier": Barrier,
    "cx": CXGate,
    "cp": CPGate,
    "swap": SwapGate,
    "rxx": RXXGate,
    "ryy": RYYGate,
    "rzz": RZZGate,
    "cy": CYGate,
    "cz": CZGate,
    "cs": CSGate,
    "ct": CTGate,
    "xy": XYResonance,
    "ccx": ToffoliGate,
    "cswap": FredkinGate,
    "mcx": MCXGate,
    "mcy": MCYGate,
    "mcz": MCZGate,
}



def p_id(p):
    ''' id : ID'''
    p[0] = p[1]


# def p_gate_like(p):
#     '''gate : id qubit_list
#             | id'('arg_list') qubit_list' 
#     '''
#     #initialize the gate
#     instruction = instructions_map[p[1]]
#     p[0] = instruction()

# def p_pulse_like(p):
    
# def p_arg(p):
#     """
#     arg : expression
#         | arg, expression
#     """
#     p[0]