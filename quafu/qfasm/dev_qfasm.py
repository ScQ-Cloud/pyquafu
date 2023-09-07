import sys
sys.path.append('C:\\Users\\AFWZSL\\Desktop\\pyquafu\\src')

import re
from collections import OrderedDict

from numpy import pi
from quafu.circuits import QuantumCircuit, QuantumRegister
from quafu.qfasm.exceptions import QfasmError

# patterns
# TODO: these regx can be split as smaller ones for 'token'
para_p = r'(-?\d+(?:\.\d+)?)\s+([a-z]+)?'  # number(float/int) + unit(e.g. pi, ns, us..)

comment_p = r'//.*'
include_p = r'include\s+"(.+)"\s*;'
openqasm_p = r'OPENQASM\s+(\d+\.\d+)\s*;'

reg_declare_p = r'(creg|qreg)\s+(\w+)\s*\[(\d+)]\s*;'
gate_declare_p = 'gate' + r'\s+(\w+)' + r'(?:\(([^)]+)\))?' + r'\s+(\w+[a-z]*(?:,\s*[a-z]+\s*)*)*\s*'

begin_scope_p = r'{'
end_scope_p = r'}\s*$'

bit_apply_p = r"(\w+)(\(.+\))?\s+(\w+\[\d+\](?:\s*,\s*\w+\[\d+\])*)\s*;"
reg_apply_p = r"(\w+)\s+(\w+(?:\s*,\s*\w+)*)\s*;"

bit_measure_p = r'measure\s+(\w+\[\d+\])\s*->\s*(\w+\[\d+\])\s*;'
reg_measure_p = r'measure\s+(\w+)\s*->\s*(\w+)\s*;'

# note: orders here is important
patterns = {'comment': comment_p,
            'reg': reg_declare_p,
            'gate': gate_declare_p,
            'bit_measure': bit_measure_p,
            'reg_measure': reg_measure_p,
            'bit_apply': bit_apply_p,
            'reg_apply': reg_apply_p,
            'OPENQASM': openqasm_p,
            'include': include_p,
            }
patterns = OrderedDict(patterns)


class QfasmPaser:
    """
    Prototype of fasm parser, which temporarily only use re to
    match patterns line by line. Later will be implemented by
    yacc and lex. # TODO
    """

    def __init__(self):
        self.in_local_scope = 0

        # TODO: store data
        self.qregs = {}
        self.cregs = {}
        self.phys_qreg = {}
        self.phys_creg = {}
        self.verbose = False

    def whether_enter_scope(self, string):
        if re.search(begin_scope_p, string):
            self.in_local_scope += 1
            return True
        return False

    def whether_exit_scope(self, string):
        if re.search(end_scope_p, string):
            self.in_local_scope -= 1
            return True
        return False
    
    def _proc_reg(self, match, qc):
        print(match)
        reg_type = match[0]
        reg_name = match[1]
        reg_size = int(match[2])

        # type-checking
        if reg_type == 'qreg':
            logic_reg, phys_reg = self.qregs, self.phys_qreg
            reg = QuantumRegister(num=reg_size, name=reg_name)
            qc.qregs.append(reg)
            # todo: qc.reg()
        elif reg_type == 'creg':
            logic_reg, phys_reg = self.cregs, self.phys_creg
            # TODO
            # reg = ClassicalRegister(num=reg_size, name=reg_name)
        else:
            raise QfasmError("unknown register type:", reg_type)

        # mapping logical-physical
        logic_reg[reg_name] = reg_size
        _num = len(phys_reg)
        _new_qubits = {reg_name + '[%d]' % i: i + _num for i in range(reg_size)}
        phys_reg.update(_new_qubits)

        # verbose
        # print('define {}: {}[{}]'.format(reg_type, reg_name, reg_size))
        # print(flush=True)

    def _proc_ins(self, match, qc):
        ins_name = match[0]
        para_str = match[1]
        if para_str:
            paras = [re.findall(para_p, para)[0] for para in para_str.split(',')]
        else:
            paras = None

        pos = [self.phys_qreg[qubit] for qubit in match[2].split(',')]

        print('apply:', ins_name)
        print('params:', paras)
        print('qubits:', pos)
        print(flush=True)

        if ins_name == "delay":
            duration = paras[0][0]
            unit = paras[0][1]
            qc.delay(pos, duration, unit)
        elif ins_name == "xy":
            duration = paras[0][0]
            unit = paras[0][1]
            qc.xy(min(pos), max(pos), duration, unit)
        else:
            units = [pi if para[0] else 1 for para in paras]
            paras = [para[0] * units[i] for i, para in enumerate(paras)]

            if ins_name in ["cx", "cy", "cz", "swap"]:
                funcs = {"cx": qc.cnot, "cy": qc.cy, "cz": qc.cz, "swap": qc.swap}
                funcs[ins_name](pos[0], pos[1])
            elif ins_name == "cp":
                qc.cp(pos[0], pos[1], paras[0])
            elif ins_name in ["rx", "ry", "rz", "p"]:
                funcs = {"rx": qc.rx, "ry": qc.ry, "rz": qc.rz, "p": qc.p}
                funcs[ins_name](pos[0], paras[0])
            elif ins_name in ["x", "y", "z", "h", "id", "s", "sdg", "t", "tdg", "sx"]:
                func = {"x": qc.x, "y": qc.y, "z": qc.z, "h": qc.h, "id": qc.id,
                        "s": qc.s, "sdg": qc.sdg, "t": qc.t, "tdg": qc.tdg, "sx": qc.sx}
                func[ins_name](pos[0])
            elif ins_name == "ccx":
                qc.toffoli(pos[0], pos[1], pos[2])
            elif ins_name == "cswap":
                qc.fredkin(pos[0], pos[1], pos[2])
            elif ins_name == "u1":
                qc.rz(pos[0], paras[0])
            elif ins_name == "u2":
                qc.rz(pos[0], paras[1])
                qc.ry(pos[0], pi / 2)
                qc.rz(pos[0], paras[0])
            elif ins_name == "u3":
                qc.rz(pos[0], paras[2])
                qc.ry(pos[0], paras[0])
                qc.rz(pos[0], paras[1])
            elif ins_name in ["rxx", "ryy", "rzz"]:
                funcs = {"rxx": qc.rxx, "ryy": qc.ryy, "rzz": qc.rzz}
                funcs[ins_name](pos[0], pos[1], paras[0])
            else:
                raise QfasmError(
                    "Operations %s may be not supported by QuantumCircuit class currently."
                    % ins_name
                )

    def match_line_pattern(self, *args):
        i, line = args
        if self.in_local_scope > 0:
            # TODO
            pass

        for p_name, pattern in patterns.items():
            match = re.findall(pattern, line)
            if match:
                return i, p_name, match[0]
        else:
            raise QfasmError("No match found in line-%d: %s" % (i, line))

    def extract_data(self, qasm: str):
        data = [self.match_line_pattern(i, line) for i, line in enumerate(qasm.splitlines())]
        # data = map(self.match_line_pattern, enumerate(qasm.splitlines()))
        return data


    def parse(self, qasm: str, verbose: bool = False) -> QuantumCircuit:
        print(qasm)
        qc = QuantumCircuit(0)

        for i, line in enumerate(qasm.splitlines()):
            line = line.strip().rstrip().lstrip()
            print('check:', line)
            if not line:
                print('ha?')
                continue
            if self.in_local_scope > 0:
                # just print at present
                # TODO: create customized gate
                _exit = self.whether_exit_scope(line)
                if verbose:
                    if _exit:
                        print(flush=True)
                    else:
                        print(' ', line)
                continue

            self.whether_enter_scope(line)

            match = re.findall(patterns['reg'], line)
            if match:
                self._proc_reg(match, qc)
                continue
            else:
                match = re.findall(patterns['gate'], line)

            if match:
                print('define gate:', match)
                if self.whether_enter_scope(line):
                    print(' ', line)
                    if self.whether_exit_scope(line):
                        print(flush=True)
                continue
            else:
                match = re.findall(patterns['comment'], line)

            if match:
                continue
            else:
                match = re.findall(patterns['bit_apply'], line)

            if match:
                print(line)
                self._proc_ins(match, qc)
                continue
            else:
                match = re.findall(patterns['bit_measure'], line)

            if match:
                q = self.phys_qreg[match[0]]
                c = self.phys_creg[match[1]]
                qc.measures[q] = c

                # TODO: check measured qubits
                # measured_qubits.append(mb)
            else:
                match = re.findall(patterns['OPENQASM'], line)

            if match:
                continue
            else:
                match = re.findall(patterns['include'], line)

            if match:
                continue
            else:
                raise QfasmError("No match found in line-%d: %s" % (i, line))

        return qc
