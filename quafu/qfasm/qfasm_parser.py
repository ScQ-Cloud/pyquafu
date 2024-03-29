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

import copy

import numpy as np
import ply.yacc as yacc
from quafu.circuits.classical_register import ClassicalRegister
from quafu.circuits.quantum_register import QuantumRegister
from quafu.elements import *
from quafu.elements.classical_element import Cif
from quafu.qfasm.exceptions import ParserError

from quafu import QuantumCircuit
from quafu.elements import Parameter, ParameterExpression
from .qfasm_lexer import QfasmLexer
from .qfasm_utils import *

unaryop = {"sin": "sin", "cos": "cos", "tan": "tan", "exp": "exp",
           "ln": "log", "sqrt": "sqrt", "acos": "arccos", "atan": "arctan", "asin": "arcsin"}
unarynp = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "ln": np.log,
    "sqrt": np.sqrt,
    "acos": np.arccos,
    "atan": np.arctan,
    "asin": np.arcsin,
}
reserved = [
    "creg",
    "qreg",
    "pi",
    "measure",
    "include",
    "barrier",
    "gate",
    "opaque",
    "reset",
    "if",
    "OPENQASM",
]


class QfasmParser(object):
    """OPENQASM2.0 Parser"""

    def __init__(self, filepath: str = None, debug=False):
        self.lexer = QfasmLexer(filepath)
        self.tokens = self.lexer.tokens
        self.precedence = (
            ("left", "+", "-"),
            ("left", "*", "/"),
            ("right", "^"),
            ("right", "UMINUS"),
        )
        self.nuop = ["barrier", "reset", "measure"]
        self.stdgate = list(gate_classes.keys())
        # extent keyword(the )
        self.stdgate.extend(["U", "CX"])
        self.mulctrl = ["mcx", "mcz", "mcy"]
        self.parser = yacc.yacc(module=self, debug=debug)
        # when there is reset/op after measure/if, set to false
        self.executable_on_backend = True
        self.has_measured = False
        self.circuit = QuantumCircuit(0)
        self.global_symtab = {}
        self.add_U_CX()
        # function argument symtab
        self.symtab = {}
        # qubit num used
        self.qnum = 0
        # cbit num used
        self.cnum = 0
        # param
        self.params = {}

    def add_U_CX(self):
        # Add U and CX in global_symtab
        U_Id = Id("U", -1, None)
        CX_Id = Id("CX", -1, None)
        UNode = SymtabNode("GATE", U_Id)
        UNode.fill_gate(qargs=[None], cargs=[None, None, None])
        CXNode = SymtabNode("GATE", CX_Id)
        CXNode.fill_gate(qargs=[None, None], cargs=[])
        self.global_symtab["U"] = UNode
        self.global_symtab["CX"] = CXNode

    # parse data
    def parse(self, data, debug=False):
        self.parser.parse(data, lexer=self.lexer, debug=debug)
        if self.circuit is None:
            raise ParserError("Exception in parser;")
        return self.circuit

    def updateSymtab(self, symtabnode: SymtabNode):
        # update Symtab
        # reg
        # print(symtabnode)
        if symtabnode.name in reserved:
            raise ParserError(f"Name cannot be reserved word:{reserved}")
        if symtabnode.is_global:
            if symtabnode.name in self.global_symtab:
                hasnode = self.global_symtab[symtabnode.name]
                raise ParserError(
                    f"Duplicate declaration for {symtabnode.name} at line {symtabnode.lineno} file {symtabnode.filename}",
                    f"First occureence at line {hasnode.lineno} file {hasnode.filename}",
                )
        else:
            # just for arg and qarg in gate declaration, so it can duplicate
            if symtabnode.name in self.symtab:
                hasnode = self.symtab[symtabnode.name]
                raise ParserError(
                    f"Duplicate declaration for {symtabnode.name} at line {symtabnode.lineno} file {symtabnode.filename}"
                )

        if symtabnode.type == "QREG":
            symtabnode.start = self.qnum
            self.qnum += symtabnode.num
            # add QuantumRegister
            if len(self.circuit.qregs) == 0:
                self.circuit.qregs.append(QuantumRegister(self.qnum, name="q"))
            else:
                self.circuit.qregs[0] = QuantumRegister(self.qnum, name="q")
        if symtabnode.type == "CREG":
            symtabnode.start = self.cnum
            self.cnum += symtabnode.num
            # add ClassicalRegister
            if len(self.circuit.cregs) == 0:
                self.circuit.cregs.append(ClassicalRegister(self.cnum, name="c"))
            else:
                self.circuit.cregs[0] = ClassicalRegister(self.cnum, name="c")

        if symtabnode.is_global:
            self.global_symtab[symtabnode.name] = symtabnode
        else:
            self.symtab[symtabnode.name] = symtabnode

    def handle_gateins(self, gateins: GateInstruction):
        gate_list = []
        # end of recurse
        if gateins.name in self.stdgate and gateins.name not in self.nuop:
            args = []
            # add qubits to args, it's might be a qubit or a qreg
            for qarg in gateins.qargs:
                if isinstance(qarg, IndexedId):
                    # check qreg's num is the same
                    if len(args) >= 1 and len(args[0]) != 1:
                        raise ParserError(
                            f"The num of qreg's qubit is inconsistent at line{gateins.lineno} file{gateins.filename}."
                        )
                    symnode = self.global_symtab[qarg.name]
                    args.append([symnode.start + qarg.num])
                elif isinstance(qarg, Id):
                    # check qreg's num is the same
                    symnode = self.global_symtab[qarg.name]
                    if len(args) >= 1 and symnode.num != len(args[0]):
                        raise ParserError(
                            f"The num of qreg's qubit is inconsistent at line{gateins.lineno} file{gateins.filename}."
                        )
                    tempargs = []
                    for i in range(symnode.num):
                        tempargs.append(symnode.start + i)
                    args.append(tempargs)
            # change carg to parameter
            for i in range(len(gateins.cargs)):
                gateins.cargs[i] = self.compute_exp(gateins.cargs[i])
            # call many times
            for i in range(len(args[0])):
                oneargs = []
                for arg in args:
                    oneargs.append(arg[i])
                # if it's U or CX
                if gateins.name == "CX":
                    gateins.name = "cx"
                    gate_list.append(gate_classes[gateins.name](*oneargs))
                elif gateins.name == "U":
                    gate_list.append(gate_classes["rz"](*[*oneargs, gateins.cargs[2]]))
                    gate_list.append(gate_classes["ry"](*[*oneargs, gateins.cargs[0]]))
                    gate_list.append(gate_classes["rz"](*[*oneargs, gateins.cargs[1]]))
                elif gateins.name in self.mulctrl:
                    gate_list.append(gate_classes[gateins.name](oneargs[:-1], oneargs[-1]))
                else:
                    # add carg to args if there is
                    if gateins.cargs is not None and len(gateins.cargs) > 0:
                        oneargs.extend(gateins.cargs)
                    gate_list.append(gate_classes[gateins.name](*oneargs))

        # if op is barrier or reset or measure
        elif gateins.name == "reset":
            for qarg in gateins.qargs:
                symnode = self.global_symtab[qarg.name]
                if isinstance(qarg, Id):
                    qlist = []
                    for i in range(symnode.num):
                        qlist.append(symnode.start + i)
                    gate_list.append(Reset(qlist))
                elif isinstance(qarg, IndexedId):
                    gate_list.append(Reset([symnode.start + qarg.num]))

        elif gateins.name == "barrier":
            qlist = []
            for qarg in gateins.qargs:
                symnode = self.global_symtab[qarg.name]
                if isinstance(qarg, Id):
                    for i in range(symnode.num):
                        qlist.append(symnode.start + i)
                elif isinstance(qarg, IndexedId):
                    qlist.append(symnode.start + qarg.num)
            gate_list.append(Barrier(qlist))

        # we have check the num of cbit and qbit
        elif gateins.name == "measure":
            bitmap = {}
            qarg = gateins.qargs[0]
            cbit = gateins.cbits[0]
            symnode = self.global_symtab[qarg.name]
            symnodec = self.global_symtab[cbit.name]
            if isinstance(qarg, Id):
                for i in range(symnode.num):
                    bitmap[symnode.start + i] = symnodec.start + i
            elif isinstance(qarg, IndexedId):
                bitmap[symnode.start + qarg.num] = symnodec.start + cbit.num
            gate_list.append(Measure(bitmap=bitmap))
            # self.circuit.measure(list(bitmap.keys()), list(bitmap.values()))
        # if it's not a gate that can be trans to circuit gate, just recurse it
        else:
            gatenode: SymtabNode = self.global_symtab[gateins.name]
            qargdict = {}
            for i in range(len(gatenode.qargs)):
                qargdict[gatenode.qargs[i].name] = gateins.qargs[i]
            cargdict = {}
            for i in range(len(gatenode.cargs)):
                cargdict[gatenode.cargs[i].name] = gateins.cargs[i]
            for ins in gatenode.instructions:
                # cannot recurse itself!
                if ins.name == gateins.name:
                    raise ParserError(
                        f"The gate {gateins.name} call itself, it's forbiddened at line {gateins.lineno} file {gateins.filename}"
                    )
                # change qarg/carg, no cbit in gate param
                # deep copy
                newins = copy.deepcopy(ins)
                # change newins's qarg to real q
                for i in range(len(newins.qargs)):
                    newins.qargs[i] = qargdict[newins.qargs[i].name]
                # change newins's carg to real carg (consider exp and parameter)
                for i in range(len(newins.cargs)):
                    # for expression and parameter, it will return parameter or int/float
                    newins.cargs[i] = self.compute_exp(newins.cargs[i], cargdict)
                # now, recurse
                gate_list.extend(self.handle_gateins(newins))

        return gate_list

    def compute_exp(self, carg, cargdict: dict={}):
        # recurse
        if isinstance(carg, int) or isinstance(carg, float) or isinstance(carg, ParameterExpression):
            return carg
        # if it's id, should get real number from gateins
        elif isinstance(carg, Id):
            if carg.name in cargdict:
                return cargdict[carg.name]
            # if it's parameter, just return
            else:
                return self.params[carg.name]
        elif isinstance(carg, UnaryExpr):
            if carg.type == "-":
                return -self.compute_exp(carg.children[0], cargdict)
            elif carg.type in unaryop:
                nowcarg = self.compute_exp(carg.children[0], cargdict)
                if isinstance(nowcarg, ParameterExpression):
                    func = getattr(nowcarg, unaryop[carg.type])
                    return func()
                else:
                    return unarynp[carg.type](nowcarg)
        elif isinstance(carg, BinaryExpr):
            cargl = self.compute_exp(carg.children[0], cargdict)
            cargr = self.compute_exp(carg.children[1], cargdict)
            if carg.type == "+":
                return cargl + cargr
            elif carg.type == "-":
                return cargl - cargr
            elif carg.type == "*":
                return cargl * cargr
            elif carg.type == "/":
                return cargl / cargr
            elif carg.type == "^":
                return cargl ** cargr

    def addInstruction(self, qc: QuantumCircuit, ins):
        if ins is None:
            return
        if isinstance(ins, GateInstruction):
            gate_list = self.handle_gateins(ins)
            for gate in gate_list:
                # print(self.circuit.num)
                qc.add_ins(gate)
                if isinstance(gate, Measure):
                    qc._measures.append(gate)
        elif isinstance(ins, IfInstruction):
            symtabnode = self.global_symtab[ins.cbits.name]
            if isinstance(ins.cbits, Id):
                cbits = []
                for i in range(symtabnode.num):
                    cbits.append(symtabnode.start + i)
            else:
                cbits = [symtabnode.start + ins.cbits.num]
            # get quantum gate
            gate_list = self.handle_gateins(ins.instruction)
            qc.add_ins(Cif(cbits=cbits, condition=ins.value, instructions=gate_list))
        else:
            raise ParserError(f"Unexpected exception when parse.")

    def check_measure_bit(self, gateins: GateInstruction):
        cbit = gateins.cbits[0]
        qarg = gateins.qargs[0]
        cbit_num = 0
        qbit_num = 0
        # check qubit
        if qarg.name not in self.global_symtab:
            raise ParserError(
                f"The qubit {qarg.name} is undefined in qubit register at line {qarg.lineno} file {qarg.filename}"
            )
        symnode = self.global_symtab[qarg.name]
        if symnode.type != "QREG":
            raise ParserError(
                f"{qarg.name} is not declared as qubit register at line {qarg.lineno} file {qarg.filename}"
            )
        if isinstance(qarg, IndexedId):
            qbit_num = 1
            if qarg.num + 1 > symnode.num:
                raise ParserError(
                    f"Qubit arrays {qarg.name} are out of bounds at line {qarg.lineno} file {qarg.filename}"
                )
        else:
            qbit_num = symnode.num
        # check cbit
        if cbit.name not in self.global_symtab:
            raise ParserError(
                f"The classical bit {cbit.name} is undefined in classical bit register at line {cbit.lineno} file {cbit.filename}"
            )
        symnode = self.global_symtab[cbit.name]
        if symnode.type != "CREG":
            raise ParserError(
                f"{cbit.name} is not declared as classical bit register at line {cbit.lineno} file {cbit.filename}"
            )
        if isinstance(cbit, IndexedId):
            cbit_num = 1
            if cbit.num + 1 > symnode.num:
                raise ParserError(
                    f"Classical bit arrays {cbit.name} are out of bounds at line {cbit.lineno} file {cbit.filename}"
                )
        else:
            cbit_num = symnode.num
        # check qubits'num matches cbits's num
        if cbit_num != qbit_num:
            raise ParserError(
                f"MEASURE: the num of qubit and clbit doesn't match at line {gateins.lineno} file {gateins.filename}"
            )

    def check_qargs(self, gateins: GateInstruction):
        # check gatename declared
        qargslist = []
        if gateins.name not in self.nuop and gateins.name not in self.mulctrl:
            if gateins.name not in self.global_symtab:
                raise ParserError(
                    f"The gate {gateins.name} is undefined at line {gateins.lineno} file {gateins.filename}"
                )
            # check if gateins.name is a gate
            gatenote = self.global_symtab[gateins.name]
            if gatenote.type != "GATE":
                raise ParserError(
                    f"The {gateins.name} is not declared as a gate at line {gateins.lineno} file {gateins.filename}"
                )
            # check args matches gate's declared args
            if len(gateins.qargs) != len(gatenote.qargs):
                raise ParserError(
                    f"The numbe of qubit declared in gate {gateins.name} is inconsistent with instruction at line {gateins.lineno} file {gateins.filename}"
                )
        # check qubits must from global symtab
        for qarg in gateins.qargs:
            if qarg.name not in self.global_symtab:
                raise ParserError(
                    f"The qubit {qarg.name} is undefined in qubit register at line {qarg.lineno} file {qarg.filename}"
                )
            symnode = self.global_symtab[qarg.name]
            if symnode.type != "QREG":
                raise ParserError(
                    f"{qarg.name} is not declared as qubit register at line {qarg.lineno} file {qarg.filename}"
                )
            # check if the qarg is out of bounds when qarg's type is indexed_id
            if isinstance(qarg, IndexedId):
                if qarg.num + 1 > symnode.num:
                    raise ParserError(
                        f"Qubit arrays {qarg.name} are out of bounds at line {qarg.lineno} file {qarg.filename}"
                    )
                qargslist.append((qarg.name, qarg.num))
            else:
                for num in range(symnode.num):
                    qargslist.append((qarg.name, num))
        # check  distinct qubits
        if len(qargslist) != len(set(qargslist)):
            raise ParserError(
                f"Qubit used as different argument when call gate {gateins.name} at line {gateins.lineno} file {gateins.filename}"
            )

    def check_param(self, carg):
        if isinstance(carg, int) or isinstance(carg, float):
            return
        elif isinstance(carg, Id) and carg.name not in self.params:
            raise ParserError(f"The parameter {carg.name} is undefined at line {carg.lineno} file {carg.filename}")
        elif isinstance(carg, UnaryExpr):
            self.check_param(carg.children[0])
        elif isinstance(carg, BinaryExpr):
            self.check_param(carg.children[0])
            self.check_param(carg.children[1])

    def check_cargs(self, gateins: GateInstruction):
        # check that cargs belongs to unary (they must be int or float or parameter)
        # cargs is different from CREG
        if gateins.name not in self.nuop and gateins.name not in self.mulctrl:
            if gateins.name not in self.global_symtab:
                raise ParserError(
                    f"The gate {gateins.name} is undefined at line {gateins.lineno} file {gateins.filename}"
                )
            gatenote = self.global_symtab[gateins.name]
            if gatenote.type != "GATE":
                raise ParserError(
                    f"The {gateins.name} is not declared as a gate at line {gateins.lineno} file {gateins.filename}"
                )
            # check every carg in [int, float, parameter]
            for carg in gateins.cargs:
                self.check_param(carg)
            # check cargs's num matches gate's delcared cargs
            if len(gateins.cargs) != len(gatenote.cargs):
                raise ParserError(
                    f"The number of classical argument declared in gate {gateins.name} is inconsistent with instruction at line {gateins.lineno} file {gateins.filename}"
                )

    def check_gate_qargs(self, gateins: GateInstruction):
        # check type and number
        qargs = gateins.qargs
        gatename = gateins.name
        qargsname = []
        if gatename != "barrier":
            # check gatename declared
            if gatename not in self.global_symtab:
                raise ParserError(
                    f"The gate {gatename} is undefined at line {gateins.lineno} file {gateins.filename}"
                )
            gatenode = self.global_symtab[gatename]
            if gatenode.type != "GATE":
                raise ParserError(
                    f"The {gatename} is not declared as a gate at line {gateins.lineno} file {gateins.filename}"
                )
            # check qarg's num matches gate's qargs, except barrier
            if len(gatenode.qargs) != len(qargs):
                raise ParserError(
                    f"The numbe of qubit declared in gate {gatename} is inconsistent with instruction at line {gateins.lineno} file {gateins.filename}"
                )
        # check gate_op's qubit args, must from gate declared argument
        for qarg in qargs:
            qargsname.append(qarg.name)
            # check qarg declaration
            if qarg.name not in self.symtab:
                raise ParserError(
                    f"The qubit {qarg.name} is undefined in gate qubit parameters at line {qarg.lineno} file {qarg.filename}"
                )
            symnode = self.symtab[qarg.name]
            if symnode.type != "QARG":
                raise ParserError(
                    f"{qarg.name} is not declared as a qubit at line {qarg.lineno} file {qarg.filename}"
                )
        if len(qargs) != len(set(qargsname)):
            raise ParserError(
                f"A qubit used as different argument when call gate {gateins.name} at line {gateins.lineno} file {gateins.filename}"
            )

    def check_gate_cargs(self, gateins: GateInstruction):
        # check gate_op's classcal args, must matches num declared by gate
        if gateins.name in ["barrier", "reset", "measure"] and len(gateins.cargs) > 0:
            raise ParserError(
                f"Barrier can not receive classical argument at line {gateins.lineno} file {gateins.filename}"
            )
        if gateins.name not in ["barrier", "reset", "measure"]:
            if gateins.name not in self.global_symtab:
                raise ParserError(
                    f"The gate {gateins.name} is undefined at line {gateins.lineno} file {gateins.filename}"
                )
            gatenode = self.global_symtab[gateins.name]
            if gatenode.type != "GATE":
                raise ParserError(
                    f"The {gateins.name} is not declared as a gate at line {gateins.lineno} file {gateins.filename}"
                )
            if len(gateins.cargs) != len(gatenode.cargs):
                raise ParserError(
                    f"The number of classical argument declared in gate {gateins.name} is inconsistent with instruction at line {gateins.lineno} file {gateins.filename}"
                )
            # check carg must from gate declared argument or int/float or parameter
            for carg in gateins.cargs:
                # recurse check expression
                self.check_carg_declartion(carg)

    def check_carg_declartion(self, node):
        if isinstance(node, int) or isinstance(node, float):
            return
        if isinstance(node, Id):
            # check declaration
            if node.name in self.symtab:
                symnode = self.symtab[node.name]
                if symnode.type != "CARG":
                    raise ParserError(
                        f"The {node.name} is not declared as a classical bit at line {node.lineno} file {node.filename}"
                    )
                return
            elif node.name not in self.params:
                raise ParserError(
                    f"The classical argument {node.name} is undefined at line {node.lineno} file {node.filename}"
                )
        if isinstance(node, UnaryExpr):
            self.check_carg_declartion(node.children[0])
        elif isinstance(node, BinaryExpr):
            for i in range(2):
                self.check_carg_declartion(node.children[i])

    start = "main"

    def p_main(self, p):
        """
        main : program
        """
        # now get the root node, return Citcuit
        self.circuit.executable_on_backend = self.executable_on_backend

    # when reduce statement into program, insert it to circuit and update symtab if it can.
    def p_program(self, p):
        """
        program : statement
        """
        p[0] = self.circuit
        self.addInstruction(p[0], p[1])

    def p_program_list(self, p):
        """
        program : program statement
        """
        p[0] = p[1]
        self.addInstruction(p[0], p[2])

        # statement |= qdecl | gatedecl | qop | if | barrier

    # statement
    # error -> ply.lex.LexToken(value)
    # others -> p -> ply.yacc.YaccProduction
    #        -> p[1] -> t.value
    def p_statement_openqasm(self, p):
        """
        statement : OPENQASM FLOAT ';'
                    | OPENQASM FLOAT error
                    | OPENQASM error
        """
        if len(p) == 3:
            raise ParserError(f"Expecting FLOAT after OPENQASM, received {p[2].value}")
        if p[3] != ";":
            raise ParserError("Expecting ';' at end of OPENQASM statement")
        if p[2] != 2.0:
            raise ParserError("Only support OPENQASM 2.0 version")
        p[0] = None

    # qop
    def p_statement_qop(self, p):
        """
        statement : qop ';'
                | qop error
                | qif ';'
                | qif error
        """
        if p[2] != ";":
            raise ParserError(f"Expecting ';' behind statement at line {p[1].lineno} file {p[1].filename}")
        p[0] = p[1]

    def p_statement_empty(self, p):
        """
        statement : ';'
        """
        p[0] = None

    def p_statement_qif(self, p):
        """
        qif : IF '(' primary MATCHES INT ')' qop
            | IF '(' primary MATCHES INT error
            | IF '(' primary MATCHES error
            | IF '(' primary error
            | IF '(' error
            | IF error
        """
        # check primary is a creg and check range
        if len(p) == 7:
            raise ParserError(
                f"Illegal IF statement, Expecting ')' at line {p[1].lineno} file {p[1].filename}"
            )
        if len(p) == 6:
            raise ParserError(
                f"Illegal IF statement, Expecting INT: the Rvalue can only be INT at line {p[1].lineno} file {p[1].filename}"
            )
        if len(p) == 5:
            raise ParserError(
                f"Illegal IF statement, Expecting '==' at line {p[1].lineno} file {p[1].filename}"
            )
        if len(p) == 4:
            raise ParserError(
                f"Illegal IF statement, Expecting Cbit: the Lvalue can only be cbit at line {p[1].lineno} file {p[1].filename}"
            )
        if len(p) == 3:
            raise ParserError(
                f"Illegal IF statement, Expecting '(' at line {p[1].lineno} file {p[1].filename}"
            )

        cbit = p[3]
        if cbit.name not in self.global_symtab:
            raise ParserError(
                f"The classical bit {cbit.name} is undefined in classical bit register at line {cbit.lineno} file {cbit.filename}"
            )
        symnode = self.global_symtab[cbit.name]
        if symnode.type != "CREG":
            raise ParserError(
                f"{cbit.name} is not declared as classical bit register at line {cbit.lineno} file {cbit.filename}"
            )
        # check range if IndexedId
        if isinstance(cbit, IndexedId):
            if cbit.num >= symnode.num:
                raise ParserError(
                    f"{cbit.name} out of range at line {cbit.lineno} file {cbit.filename}"
                )
            # optimization: If the value that creg can represent is smaller than Rvalue, just throw it
            if p[5] > 2:
                p[0] = None
            else:
                p[0] = IfInstruction(
                    node=p[1], cbits=p[3], value=p[5], instruction=p[7]
                )
        elif isinstance(cbit, Id):
            # optimization: If the value that creg can represent is smaller than Rvalue, just throw it
            num = symnode.num
            if pow(2, num) - 1 < p[5]:
                p[0] = None
            else:
                p[0] = IfInstruction(
                    node=p[1], cbits=p[3], value=p[5], instruction=p[7]
                )
        self.executable_on_backend = False

    def p_unitaryop(self, p):
        """
        qop : id primary_list
            | id '(' ')' primary_list
            | id '(' expression_list ')' primary_list
        """
        # return circuit gate instance
        if len(p) == 5:
            p[0] = GateInstruction(node=p[1], qargs=p[4], cargs=[])
        if len(p) == 3:
            p[0] = GateInstruction(node=p[1], qargs=p[2], cargs=[])
        if len(p) == 6:
            p[0] = GateInstruction(node=p[1], qargs=p[5], cargs=p[3])
        # check args
        self.check_qargs(p[0])
        self.check_cargs(p[0])
        if self.has_measured:
            self.executable_on_backend = False

    def p_unitaryop_error(self, p):
        """
        qop : id '(' ')' error
            | id '(' error
            | id '(' expression_list ')' error
            | id '(' expression_list error
        """
        if len(p) == 4 or (len(p) == 5 and p[4] != ")"):
            raise ParserError(
                f"Expecting ')' after '(' at line {p[1].lineno} file {p[1].filename}"
            )
        raise ParserError(
            f"Expecting qubit list, received {p[len(p)-1].value} at line {p[1].lineno} file {p[1].filename}"
        )

    # measure
    def p_measure(self, p):
        """
        qop : MEASURE primary ASSIGN primary
        """
        # check and return gateInstruction
        p[0] = GateInstruction(node=p[1], qargs=[p[2]], cargs=[], cbits=[p[4]])
        self.check_measure_bit(p[0])
        self.has_measured = True

    def p_measure_error(self, p):
        """
        qop : MEASURE primary ASSIGN error
            | MEASURE primary error
            | MEASURE error
        """
        if len(p) == 5:
            raise ParserError(
                f"Expecting qubit or qubit register after '->' at line {p[1].lineno} file {p[1].filename}"
            )
        if len(p) == 4:
            raise ParserError(
                f"Expecting '->' for MEASURE at line {p[1].lineno} file {p[1].filename}"
            )
        if len(p) == 3:
            raise ParserError(
                f"Expecting qubit or qubit register after 'measure' at line {p[1].lineno} file {p[1].filename}"
            )

    # barrier
    def p_barrier(self, p):
        """
        qop : BARRIER primary_list
        """
        # check and return gateInstruction
        p[0] = GateInstruction(node=p[1], qargs=p[2], cargs=[])
        self.check_qargs(p[0])

    def p_barrier_error(self, p):
        """
        qop : BARRIER error
        """
        raise ParserError(
            f"Expecting Qubit:BARRIER only opperate qubits at line {p[1].lineno} file {p[1].filename}"
        )

    # reset
    def p_reset(self, p):
        """
        qop : RESET primary
        """
        p[0] = GateInstruction(node=p[1], qargs=[p[2]], cargs=[])
        self.check_qargs(p[0])
        self.executable_on_backend = False

    def p_reset_error(self, p):
        """
        qop : RESET error
        """
        raise ParserError(
            f"Expecting Qubit: RESET only opperate qubit at line {p[1].lineno} file {p[1].filename}"
        )

    # gate_qarg_list
    def p_gate_qarg_list_begin(self, p):
        """
        qarg_list : id
        """
        p[0] = [p[1]]
        newsymtabnode = SymtabNode("QARG", p[1], False, True)
        self.updateSymtab(newsymtabnode)

    def p_gate_qarg_list_next(self, p):
        """
        qarg_list : qarg_list ',' id
        """
        p[0] = p[1]
        p[0].append(p[3])
        newsymtabnode = SymtabNode("QARG", p[3], False, True)
        self.updateSymtab(newsymtabnode)

    # gate_carg_list
    def p_gate_carg_list_begin(self, p):
        """
        carg_list : id
        """
        p[0] = [p[1]]
        newsymtabnode = SymtabNode("CARG", p[1], False)
        self.updateSymtab(newsymtabnode)

    def p_gate_carg_list_next(self, p):
        """
        carg_list : carg_list ',' id
        """
        p[0] = p[1]
        p[0].append(p[3])
        newsymtabnode = SymtabNode("CARG", p[3], False)
        self.updateSymtab(newsymtabnode)

    # gatedecl
    def p_statement_gatedecl_nolr(self, p):
        """
        statement : GATE id gate_scope qarg_list gate_body
                | GATE error
                | GATE id gate_scope error
                | GATE id gate_scope qarg_list error
        """
        if len(p) == 3:
            raise ParserError(
                f"Expecting ID after 'gate', received {p[len(p)-1].value} at line {p[1].lineno} file {p[1].filename}"
            )
        if len(p) == 5:
            raise ParserError(
                f"Expecting '(' or qubit list after gate name, received {p[len(p)-1].value} at line {p[1].lineno} file {p[1].filename}"
            )
        if len(p) == 6 and not isinstance(p[5], list):
            raise ParserError(
                f"Expecting gate body qubit list, received {p[len(p)-1].value} at line {p[1].lineno} file {p[1].filename}"
            )
        newsymtabnode = SymtabNode("GATE", p[2])
        newsymtabnode.fill_gate(p[4], p[5])
        self.updateSymtab(newsymtabnode)

    def p_statement_gatedecl_noargs(self, p):
        """
        statement : GATE id gate_scope '(' ')' qarg_list gate_body
                | GATE id gate_scope '(' error
                | GATE id gate_scope '(' ')' error
                | GATE id gate_scope '(' ')' qarg_list error
        """
        if len(p) == 6:
            raise ParserError(
                f"Expecting ')' or argument list after '(', received {p[len(p)-1].value} at line {p[1].lineno} file {p[1].filename}"
            )
        if len(p) == 7:
            raise ParserError(
                f"Expecting qubit list after ')', received {p[len(p)-1].value} at line {p[1].lineno} file {p[1].filename}"
            )
        if len(p) == 8 and not isinstance(p[7], list):
            raise ParserError(
                f"Expecting gate body after qubit list, received {p[len(p)-1].value} at line {p[1].lineno} file {p[1].filename}"
            )
        newsymtabnode = SymtabNode("GATE", p[2])
        newsymtabnode.fill_gate(p[6], p[7])
        self.updateSymtab(newsymtabnode)

    def p_statement_gatedecl_args(self, p):
        """
        statement : GATE id gate_scope '(' carg_list ')' qarg_list gate_body
                | GATE id gate_scope '(' carg_list ')' qarg_list error
                | GATE id gate_scope '(' carg_list ')' error
                | GATE id gate_scope '(' carg_list error
        """
        if len(p) == 7:
            raise ParserError(
                f"Expecting ')' after argument list, received {p[len(p)-1].value} at line {p[1].lineno} file {p[1].filename}"
            )
        if len(p) == 8:
            raise ParserError(
                f"Expecting qubit list after ')', received {p[len(p)-1].value} at line {p[1].lineno} file {p[1].filename}"
            )
        if len(p) == 9 and not isinstance(p[8], list):
            raise ParserError(
                f"Expecting gate body after qubit list, received {p[len(p)-1].value} at line {p[1].lineno} file {p[1].filename}"
            )
        if len(p) != 9:
            raise ParserError(f"Invaild GATE statement")
        newsymtabnode = SymtabNode("GATE", p[2])
        newsymtabnode.fill_gate(p[7], p[8], p[5])
        self.updateSymtab(newsymtabnode)

    def p_gate_scope(self, _):
        """
        gate_scope :
        """
        self.symtab = {}

    # gatebody
    def p_gate_body_emptybody(self, p):
        """
        gate_body : '{' gate_scope '}'
                    | '{' gate_scope error
        """
        if p[3] != "}":
            raise ParserError(
                "Expecting '}' at the end of gate definition; received " + p[3].value
            )
        p[0] = []

    def p_gate_body(self, p):
        """
        gate_body : '{' gop_list gate_scope '}'
                    | '{' gop_list gate_scope error
        """
        if p[4] != "}":
            raise ParserError(
                "Expecting '}' at the end of gate definition; received " + p[4].value
            )
        p[0] = p[2]

    def p_gop_list_begin(self, p):
        """
        gop_list : gop
        """
        p[0] = [p[1]]

    def p_gop_list_next(self, p):
        """
        gop_list : gop_list gop
        """
        p[0] = p[1]
        p[0].append(p[2])

    # gop
    # CX | U | ID(cargs)qargs | reset |
    def p_gop_nocargs(self, p):
        """
        gop : id id_list ';'
            | id id_list error
            | id '(' ')' id_list ';'
            | id '(' ')' id_list error
            | id '(' ')' error
            | id '(' error
        """
        if len(p) == 4 and p[2] == "(":
            raise ParserError(
                f"Expecting ')' for gate {p[1].name} at line {p[1].lineno} file {p[1].filename}"
            )
        if len(p) == 4 and p[3] != ";":
            raise ParserError(
                f"Expecting ';' after gate {p[1].name} at line {p[1].lineno} file {p[1].filename}"
            )
        if len(p) == 6 and p[5] != ";":
            raise ParserError(
                f"Expecting ';' after gate {p[1].name} at line {p[1].lineno} file {p[1].filename}"
            )
        if len(p) == 5:
            raise ParserError(
                f"Expecting ID: Invalid qubit list for gate {p[1].name} at line {p[1].lineno} file {p[1].filename}"
            )
        qargs = p[2] if len(p) == 4 else p[4]
        p[0] = GateInstruction(node=p[1], qargs=qargs, cargs=[])
        self.check_gate_qargs(p[0])
        self.check_gate_cargs(p[0])

    def p_gop_cargs(self, p):
        """
        gop : id '(' expression_list ')' id_list ';'
            | id '(' expression_list ')' id_list error
            | id '(' expression_list ')' error
            | id '(' expression_list error
        """
        if len(p) == 7 and p[6] != ";":
            raise ParserError(
                f"Expecting ';' after gate {p[1].name} at line {p[1].lineno}"
            )
        if len(p) == 6:
            raise ParserError(
                f"Expecting qubit id after gate {p[1].name} at line {p[1].lineno}"
            )
        if len(p) == 5:
            raise ParserError(
                f"Expecting ')' after gate {p[1].name} at line {p[1].lineno}"
            )
        p[0] = GateInstruction(node=p[1], qargs=p[5], cargs=p[3])
        # check qubit
        self.check_gate_qargs(p[0])
        # check expression_list
        self.check_gate_cargs(p[0])

    def p_gop_barrier(self, p):
        """
        gop : BARRIER id_list ';'
            | BARRIER id_list error
            | BARRIER error
        """
        if len(p) == 3:
            raise ParserError(
                f"Expecting ID: Invalid barrier qubit list inside gate definition, at line {p[1].lineno} file {p[1].filename}"
            )
        if len(p) == 4 and p[3] != ";":
            raise ParserError(
                f"Expecting ';' after barrier at line {p[1].lineno} file {p[1].filename}"
            )
        p[0] = GateInstruction(node=p[1], qargs=p[2], cargs=[])
        self.check_gate_qargs(p[0])

    # regdecl
    def p_statement_bitdecl(self, p):
        """
        statement : qdecl ';'
                    | cdecl ';'
                    | defparam ';'
                    | defparam error
                    | qdecl error
                    | cdecl error
                    | error
        """
        if len(p) == 2:
            raise ParserError(f"Expecting valid statement")
        if p[2] != ";":
            raise ParserError(
                f"Expecting ';' in qreg or creg declaration at line {p.lineno(2)}"
            )
        p[0] = p[1]

    def p_statement_defparam(self, p):
        """
        defparam : id EQUAL FLOAT
                 | id EQUAL INT
                 | id EQUAL error
        """
        if not isinstance(p[3], int) and not isinstance(p[3], float):
            raise ParserError(f"Expecting 'INT' or 'FLOAT behind '=' at line {p[1].lineno} file {p[1].filename}")
        param_name = p[1].name
        if param_name in self.params:
            raise ParserError(f"Duplicate declaration for parameter {p[1].name} at line {p[1].lineno} file {p[1].filename}")
        self.params[param_name] = Parameter(param_name, p[3])
        p[0] = None

    def p_qdecl(self, p):
        """
        qdecl : QREG indexed_id
                | QREG error
        """
        if not isinstance(p[2], IndexedId):
            raise ParserError(
                f"Expecting ID[int] after QREG at line {p[1].lineno} file {p[1].filename}, received {p[2].value}"
            )
        if p[2].num <= 0:
            raise ParserError(
                f"QREG size must be positive at line {p[2].lineno} file {p[2].filename}"
            )
        newsymtabnode = SymtabNode("QREG", p[2], True, True)
        self.updateSymtab(newsymtabnode)
        p[0] = None

    def p_cdecl(self, p):
        """
        cdecl : CREG indexed_id
                | CREG error
        """
        if not isinstance(p[2], IndexedId):
            raise ParserError(
                f"Expecting ID[int] after CREG at line {p[1].lineno} file {p[1].filename}, received {p[2].value}"
            )
        if p[2].num <= 0:
            raise ParserError(
                f"CREG size must be positive at line {p[2].lineno} file {p[2].filename}"
            )
        newsymtabnode = SymtabNode("CREG", p[2], True, False)
        self.updateSymtab(newsymtabnode)
        p[0] = None

    # id
    def p_id(self, p):
        """
        id : ID
            | error
        """
        # It's instance of Id class, passed from t.value
        if not isinstance(p[1], Id):
            raise ParserError(f"Expecting an ID, received {str(p[1].value)}")
        p[0] = p[1]

    # indexed_id
    def p_indexed_id(self, p):
        """
        indexed_id : id '[' INT ']'
                    | id '[' INT error
                    | id '[' error
        """
        if len(p) == 4 or (len(p) == 5 and p[4] != "]"):
            raise ParserError(
                f"Expecting INT after [, received{str(p[len(p)-1].value)}"
            )
        if len(p) == 5 and p[4] == "]":
            p[0] = IndexedId(p[1], p[3])

    # primary
    # only q or q[], used for U CX measure reset
    def p_primary(self, p):
        """
        primary : id
                | indexed_id
        """
        p[0] = p[1]

    # primary_list
    # the anylist in bnf
    def p_primary_list(self, p):
        """
        primary_list : primary
                     | primary_list ',' primary
        """
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1]
            p[0].append(p[3])

    # id_list
    # for decl gate
    def p_id_list_begin(self, p):
        """
        id_list : id
        """
        p[0] = [p[1]]

    def p_id_list_next(self, p):
        """
        id_list : id_list ',' id
        """
        p[0] = p[1]
        p[0].append(p[3])

    # unary
    def p_unary_int(self, p):
        """
        unary : INT
        """
        p[0] = int(p[1])

    def p_unary_float(self, p):
        """
        unary : FLOAT
        """
        p[0] = float(p[1])

    def p_unary_pi(self, p):
        """
        unary : PI
        """
        p[0] = np.pi

    # id from ID
    def p_unary_id(self, p):
        """
        unary : id
        """
        p[0] = p[1]

    # expr
    def p_expr_binary(self, p):
        """
        expression : expression '*' expression
                    | expression '/' expression
                    | expression '+' expression
                    | expression '-' expression
                    | expression '^' expression
        """
        if p[2] == "/" and p[3] == 0:
            raise ParserError(
                f"Divided by 0 at line {self.lexer.lexer.lineno} file {self.lexer.lexer.filename}"
            )
        if isinstance(p[1], Node) or isinstance(p[3], Node):
            p[0] = BinaryExpr(p[2], p[1], p[3])
        else:
            # int or float
            if p[2] == "*":
                p[0] = p[1] * p[3]
            elif p[2] == "/":
                p[0] = p[1] / p[3]
            elif p[2] == "^":
                p[0] = p[1] ** p[3]
            elif p[2] == "+":
                p[0] = p[1] + p[3]
            elif p[2] == "-":
                p[0] = p[1] - p[3]

    def p_expr_uminus(self, p):
        """
        expression : - expression %prec UMINUS
        """
        if isinstance(p[2], Node):
            p[0] = UnaryExpr("-", p[2])
        else:
            # int or float
            p[0] = -p[2]

    def p_expr_unary(self, p):
        """
        expression : unary
        """
        p[0] = p[1]

    def p_expr_pare(self, p):
        """
        expression : '(' expression ')'
        """
        p[0] = p[2]

    def p_expr_mathfunc(self, p):
        """
        expression : id '(' expression ')'
        """
        if p[1].name not in unaryop:
            raise ParserError(
                f"Math function {p[1].name} not supported, only support {unaryop.keys()} line {p[1].lineno} file {p[1].filename}"
            )
        if not isinstance(p[3], Node):
            p[0] = unarynp[p[1].name](p[3])
        else:
            p[0] = UnaryExpr(p[1].name, p[3])

    # Exprlist
    def p_exprlist_begin(self, p):
        """
        expression_list : expression
        """
        p[0] = [p[1]]

    def p_exprlist_next(self, p):
        """
        expression_list : expression_list ',' expression
        """
        p[0] = p[1]
        p[0].append(p[3])

    # Only filename provide string
    # So, It will never get a string in parser
    def p_ignore(self, _):
        """
        ignore : STRING
        """
        pass

    def p_empty(self, p):
        """
        empty :
        """
        pass

    def p_error(self, p):
        # EOF case
        if p is None or self.lexer.lexer.token() is None:
            raise ParserError("Error at end of file")
        print(
            f"Error near line {self.lexer.lexer.lineno}, Column {self.cal_column(self.lexer.data, p)}"
        )

    def cal_column(self, data: str, p):
        "Compute the column"
        begin_of_line = data.rfind("\n", 0, p.lexpos)
        begin_of_line = max(0, begin_of_line)
        column = p.lexpos - begin_of_line + 1
        return column
