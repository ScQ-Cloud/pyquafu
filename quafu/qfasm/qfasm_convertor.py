from typing import List
from quafu.circuits.quantum_circuit import QuantumCircuit
from .qfasm_parser import QfasmParser
from .exceptions import ParserError

def qasm_to_quafu(openqasm: str=None, filepath: str=None):
    """
    Initialize pyquafu circuit from openqasm text, mainly by
    utilizing regular expressions. This function is developed
    not only to be backward compatible with pure quantum part of
    OPENQASM2.0, but also to support provide foundation of more
    powerful syntax parser.
    """
    if filepath is None and openqasm is None:
        raise ParserError("Please provide a qasm str or a qasm file.")
    if filepath:
        with open(filepath) as ifile:
            openqasm = ifile.read()
    qparser = QfasmParser(filepath)
    qc = qparser.parse(openqasm)
    qc.openqasm = openqasm
    if not qc.executable_on_backend:
        print(
            "Warning: At present, quafu's back-end real machine does not support dynamic circuits."
        )
    return qc

def qasm2_to_quafu_qc(qc: QuantumCircuit, openqasm: str=None, filepath: str=None):
    """
    Initialize pyquafu circuit from openqasm text, mainly by
    utilizing regular expressions. This function is developed
    not only to be backward compatible with pure quantum part of
    OPENQASM2.0, but also to support provide foundation of more
    powerful syntax parser.
    """
    if filepath is None and openqasm is None:
        raise ParserError("Please provide a qasm str or a qasm file.")
    if filepath:
        with open(filepath) as ifile:
            openqasm = ifile.read()
    qparser = QfasmParser(filepath)
    qc = qparser.parse(openqasm)
    qc.openqasm = openqasm
    if not qc.executable_on_backend:
        print(
            "Warning: At present, quafu's back-end real machine does not support dynamic circuits."
        )

if __name__ == '__main__':
    pass
