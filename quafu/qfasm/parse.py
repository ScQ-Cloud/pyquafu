from qfasm_parser import QfasmParser
from exceptions import ParserError
# parse file and parse data

def parse_str(filename:str=None, data:str=None):
    if filename is None and data is None:
        raise ParserError("Please provide a qasm str or a qasm file.")
    if filename:
        with open(filename) as ifile:
            data = ifile.read()
    
    qasm = QfasmParser(filename)
    return qasm.parse(data)

if __name__ == "__main__":
    from parse import parse_str
    qc = parse_str(filename='qasm.qasm')

    print(qc.gates)