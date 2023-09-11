from parse import parse_str
from quafu.circuits.quantum_circuit import QuantumCircuit

def qasm_to_circuit(qasm):
    qc = parse_str(data = qasm)
    return qc


def qasm2_to_quafu_qc(qc: QuantumCircuit, openqasm: str):
    """
    Initialize pyquafu circuit from openqasm text, mainly by
    utilizing regular expressions. This function is developed
    not only to be backward compatible with pure quantum part of
    OPENQASM2.0, but also to support provide foundation of more
    powerful syntax parser.
    """
    qc = parse_str(data = openqasm)
    return qc
    


if __name__ == '__main__':
    import re

    pattern = r"[a-z]"

    text = "Hello, world! This is a test."

    matches = re.findall(pattern, text)

    print(matches)
