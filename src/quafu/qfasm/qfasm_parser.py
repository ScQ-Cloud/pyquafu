import ply.yacc as yacc
from qfasm_utils import *
from quafu.qfasm.exceptions import ParserError

from .qfasmlex import QfasmLexer
import numpy as np
from quafu import QuantumCircuit

# global symtab
global_symtab = {}

# Add U and CX in global_symtab
U_Id = Id('U', -1, None)
CX_Id = Id('CX', -1, None)
UNode = SymtabNode('GATE', U_Id)
UNode.fill_gate(qargs=[None], cargs=[None,None,None])
CXNode = SymtabNode('GATE', CX_Id)
CXNode.fill_gate(qargs=[None,None],cargs=[])
global_symtab['U'] = UNode
global_symtab['CX'] = CXNode

# function argument symtab
symtab = {}
# qubit num used
qnum = 0
# cbit num used
cnum = 0


unaryop = ["sin", "cos", "tan", "exp", "ln", "sqrt", "acos", "atan", "asin"]
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

def updateSymtab(symtabnode:Node):
        # update Symtab
        # reg 
        global symtab
        global global_symtab
        if symtabnode.is_global:
            if symtabnode.name in global_symtab:
                hasnode = global_symtab[symtabnode.name]
                raise ParserError(f"Duplicate declaration for {symtabnode.name} at line {symtabnode.lineno} file {symtabnode.filename}",
                                  f"First occureence at line {hasnode.lineno} file {hasnode.filename}")
        else:
            # just for arg and qarg in gate declaration, so it can duplicate
            if symtabnode.name in symtab:
                hasnode = symtab[symtabnode.name]
                raise ParserError(f"Duplicate declaration for {symtabnode.name} at line {symtabnode.lineno} file {symtabnode.filename}")

        if symtabnode.type == 'QREG':
            symtabnode.start = qnum
            qnum += symtabnode.num

        if symtabnode.type == 'CREG':
            symtabnode.start = cnum
            cnum += symtabnode.num

        if symtabnode.is_global:
            global_symtab[symtabnode.name] = symtabnode
        else:
            symtab[symtabnode.name] = symtabnode


# 从最底层向上归约写
class QfasmParser(object):
    """OPENQASM2.0 Parser"""
    def __init__(self, filename,debug=False):
        self.lexer = QfasmLexer(filename)
        self.tokens = self.lexer.tokens
        self.precedence = (
            ('left', '+', '-'),
            ('left', '*', '/'),
            ("left", "^"),
            ("right","UMINUS")
        )
        self.nuop = ['barrier', 'reset', 'measure']
        self.stdgate = gate_classes.keys()
        self.parser = yacc.yacc(module=self, debug=debug)

    def cal_column(self, data: str, p):
        "Compute the column"
        begin_of_line = data.rfind('\n', 0, p.lexpos)
        begin_of_line = max(0, begin_of_line)
        column = p.lexpos - begin_of_line + 1
        return column
    
    def addInstruction(self, qc, statement):
        pass
    
    def check_measure_bit(self, gateins:GateInstruction):
        global global_symtab
        cbit = gateins.cbits[0]
        qarg = gateins.qargs[0]
        cbit_num = 0
        qbit_num = 0
        # check qubit
        if qarg.name not in global_symtab:
            raise ParserError(f"The qubit {qarg.name} is undefined in qubit register at line {qarg.lineno} file {qarg.filename}")
        symnode = global_symtab[qarg.name]
        if symnode.type != 'QREG':
                raise ParserError(f"{qarg.name} is not declared as qubit register at line {qarg.lineno} file {qarg.filename}")
        if isinstance(qarg, IndexedId):
                qbit_num = 1
                if qarg.num + 1 > symnode.num:
                    raise ParserError(f"Qubit arrays {qarg.name} are out of bounds at line {qarg.lineno} file {qarg.filename}")
        else:
            qbit_num = symnode.num
        # check cbit
        if cbit.name not in global_symtab:
            raise ParserError(f"The classical bit {cbit.name} is undefined in classical bit register at line {cbit.lineno} file {cbit.filename}")
        symnode = global_symtab[cbit.name]
        if symnode.type != 'CREG':
                raise ParserError(f"{cbit.name} is not declared as classical bit register at line {cbit.lineno} file {cbit.filename}")
        if isinstance(cbit, IndexedId):
                cbit_num = 1
                if cbit.num + 1 > symnode.num:
                    raise ParserError(f"Classical bit arrays {cbit.name} are out of bounds at line {cbit.lineno} file {cbit.filename}")
        else:
            cbit_num = symnode.num
        # check qubits'num matches cbits's num
        if cbit_num != qbit_num:
            raise ParserError(f"MEASURE: the num of qubit and clbit doesn's match at line {gateins.lineno} file {gateins.filename}")



    def check_qargs(self, gateins:GateInstruction):
        # check gatename declared
        global global_symtab
        qargslist = []
        if gateins.name not in self.nuop:
            if gateins.name not in global_symtab:
                raise ParserError(f"The gate {gateins.name} is undefined at line {gateins.lineno} file {gateins.filename}")
            # check if gateins.name is a gate
            gatenote = global_symtab[gateins.name]
            if gatenote.type != 'GATE':
                raise ParserError(f"The {gateins.name} is not defined as a gate at line {gateins.lineno} file {gateins.filename}")
            # check args matches gate's declared args
            if len(gateins.qargs) != len(gatenote.qargs):
                 raise ParserError(f"The numbe of qubit declared in gate {gateins.name} is inconsistent with instruction at line {gateins.lineno} file {gateins.filename}")
        # check qubits must from global symtab
        for qarg in gateins.qargs:
            if qarg.name not in global_symtab:
                raise ParserError(f"The qubit {qarg.name} is undefined in qubit register at line {qarg.lineno} file {qarg.filename}")
            symnode = global_symtab[qarg.name]
            if symnode.type != 'QREG':
                raise ParserError(f"{qarg.name} is not declared as qubit register at line {qarg.lineno} file {qarg.filename}")
            # check if the qarg is out of bounds when qarg's type is indexed_id 
            if isinstance(qarg, IndexedId):
                if qarg.num + 1 > symnode.num:
                    raise ParserError(f"Qubit arrays {qarg.name} are out of bounds at line {qarg.lineno} file {qarg.filename}")
                qargslist.append((qarg.name, qarg.num))
            else:
                for num in symnode.num:
                    qargslist.append((qarg.name, num))
        # check  distinct qubits
        if len(qargslist) != len(set(qargslist)):
            raise ParserError(f"Qubit used as different argument when call gate {gateins.name} at line {gateins.lineno} file {gateins.filename}")
    
    def check_cargs(self, gateins:GateInstruction):
        # check that cargs belongs to unary (they must be int or float)
        # cargs is different from CREG
        if gateins.name not in self.nuop:
            if gateins.name not in global_symtab:
                raise ParserError(f"The gate {gateins.name} is undefined at line {gateins.lineno} file {gateins.filename}")
            gatenote = global_symtab[gateins.name]
            if gatenote.type != 'GATE':
                raise ParserError(f"The {gateins.name} is not defined as a gate at line {gateins.lineno} file {gateins.filename}")
            # check every carg in [int, float]
            for carg in gateins.cargs:
                if not (isinstance(carg, int) or isinstance(carg, float)): 
                    raise ParserError(f"Classical argument must be of type int or float at line {gateins.lineno} file {gateins.filename}")
            # check cargs's num matches gate's delcared cargs
            if len(gateins.cargs) != len(gatenote.cargs):
               raise ParserError(f"The number of classical argument declared in gate {gateins.name} is different from instruction at line {gateins.lineno} file {gateins.filename}") 

    
    def check_gate_qargs(self, gateins:GateInstruction):
        # check type and number
        global symtab
        global global_symtab
        qargs = gateins.qargs
        gatename = gateins.name
        qargsname = []
        if gatename != 'barrier':
            # check gatename declared
            if gatename not in global_symtab:
                raise ParserError(f"The gate {gatename} is undefined at line {gateins.lineno} file {gateins.filename}")
            gatenode = global_symtab[gatename]
            if gatenode.type != "GATE":
                raise ParserError(f"The {gatename} is not defined as a gate at line {gateins.lineno} file {gateins.filename}")
            # check qarg's num matches gate's qargs, except barrier
            if len(gatenode.qargs) != len(qargs):
                raise ParserError(f"The numbe of qubit declared in gate {gatename} is inconsistent with instruction at line {gateins.lineno} file {gateins.filename}")
        # check gate_op's qubit args, must from gate declared argument
        for qarg in qargs:
            qargsname.append(qarg.name)
            # check qarg declaration
            if qarg.name not in symtab:
                raise ParserError(f"The qubit {qarg.name} is undefined in gate qubit parameters at line {qarg.lineno} file {qarg.filename}")
            symnode = symtab[qarg.name]
            if symnode.type != 'QARG':
                raise ParserError(f"{qarg.name} is not declared as a qubit at line {qarg.lineno} file {qarg.filename}")
        if len(qargs) != len(set(qargsname)):
            raise ParserError(f"A qubit used as different argument when call gate {gateins.name} at line {gateins.lineno} file {gateins.filename}")

    def check_gate_cargs(self, gateins:GateInstruction):
        # check gate_op's classcal args, must matches num declared by gate
        global global_symtab
        if gateins.name == 'barrier' and len(gateins.cargs) > 0:
            raise ParserError(f"Barrier can not receive classical argument at line {gateins.lineno} file {gateins.filename}")
        if gateins.name != 'barrier':
            if gateins.name not in global_symtab:
                raise ParserError(f"The gate {gateins.name} is undefined at line {gateins.lineno} file {gateins.filename}")
            gatenode = global_symtab[gateins.name]
            if gatenode.type != "GATE":
                raise ParserError(f"The {gateins.name} is not defined as a gate at line {gateins.lineno} file {gateins.filename}")
            if len(gateins.cargs) != len(gatenode.cargs):
                raise ParserError(f"The number of classical argument declared in gate {gateins.name} is different from instruction at line {gateins.lineno} file {gateins.filename}")
        
    start = "main"

    def p_main(self, p):
        """
        main : program
        """
        # now get the root node, return Citcuit
        self.circuit = p[1]

    # when reduce statement into program, insert it to circuit and update symtab if it can.
    def p_program(self, p):
        """
        program : statement
        """
        p[0] = QuantumCircuit(0)
        self.addInstruction(p[0], p[1])
        
    def p_program_list(self, p):
        """
        program : program statement
        """    
        p[0] = p[1]
        self.addInstruction(p[0],p[1])

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
            raise ParserError(f"Expected FLOAT after OPENQASM, received {p[2].value}")
        if p[3] != ';':
            raise ParserError("Missing ';' at end of OPENQASM statement")
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
        if p[2] != ';':
            raise ParserError(f"Missing ';' behind statement")
        p[0] = p[1]
        self.addInstruction(p[0])
    
    def p_statement_qif(self, p):
        """
        qif : IF '(' id MATCHES INT ')' qop 
                |   IF '(' id MATCHES INT ')' error
                |   IF '(' id MATCHES INT error
                |   IF '(' id MATCHES error
                |   IF '(' id error
                |   IF error
        """
        # check id is a creg and check range
        pass
    
    def p_unitaryop(self, p):
        """
        qop : id primary_list
            | id '(' ')' primary_list 
            | id '(' expression_list ')' primary_list
        """
        # return circuit gate instance 
        if len(p) == 5:
            p[0] = GateInstruction(node=p[1], qargs=p[2], cargs=[])
        if len(p) == 3:
            p[0] = GateInstruction(node=p[1], qargs=p[2], cargs=[])
        if len(p) == 6:
            p[0] = GateInstruction(node=p[1], qargs=p[5], cargs=p[3])
        # check args
        self.check_qargs(p[0])
        self.check_cargs(p[0])

    def p_unitaryop_error(self, p):
        """
        qop : id '(' ')' error 
            | id '(' error
            | id '(' expression_list ')' error
            | id '(' expression_list error
        """
        if len(p) == 4 or (len(p) == 5 and p[4] != ')'):
            raise ParserError(f"Missing ')' after '(' at line {p[1].lineno} file {p[1].filename}")
        raise ParserError(f"Expecting qubit list, received {p[len(p)-1].value} at line {p[1].lineno} file {p[1].filename}")

    # measure
    def p_measure(self, p):
        """
        qop : MEASURE primary ASSIGN primary
        """
        # check and return gateInstruction
        p[0] = GateInstruction(node=p[1], qargs=[p[2]], cbits=[p[4]])
        self.check_measure_bit(p[0])

    def p_measure_error(self, p):
        """
        qop : MEASURE primary ASSIGN error
            | MEASURE primary error
            | MEASURE error
        """
        if len(p) == 5:
            raise ParserError(f"Expecting qubit or qubit register after '->' at line {p[1].lineno} file {p[1].filename}")
        if len(p) == 4:
            raise ParserError(f"Expecting '->' for MEASURE at line {p[1].lineno} file {p[1].filename}")
        if len(p) == 3:
            raise ParserError(f"Expecting qubit or qubit register after 'measure' at line {p[1].lineno} file {p[1].filename}")

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
        raise ParserError(f"BARRIER only opperate qubits at line {p[1].lineno} file {p[1].filename}")

    # reset
    def p_reset(self, p):
        """
        qop : RESET primary
        """
        p[0] = GateInstruction(node=p[1], qargs=[p[2]], cargs=[])
        self.check_qargs(p[0])
    
    def p_reset(self, p):
        """
        qop : RESET error
        """
        raise ParserError(f"RESET only opperate qubit at line {p[1].lineno} file {p[1].filename}")
    
    # gate_qarg_list
    def p_gate_qarg_list_begin(self, p):
        """
        arg_list : id
        """
        p[0] = [p[1]]
        newsymtabnode = SymtabNode('QARG', p[1], False, True)
        updateSymtab(newsymtabnode)

    def p_gate_qarg_list_next(self, p):
        """
        qarg_list : qarg_list ',' id
        """
        p[0] = p[1]
        p[0].append(p[3])
        newsymtabnode = SymtabNode('QARG', p[3], False, True)
        updateSymtab(newsymtabnode)

    # gate_carg_list
    def p_gate_carg_list_begin(self, p):
        """
        carg_list : id
        """
        p[0] = [p[1]]
        newsymtabnode = SymtabNode('CARG', p[1], False)
        updateSymtab(newsymtabnode)

    def p_gate_carg_list_next(self, p):
        """
        carg_list : carg_list ',' id
        """
        p[0] = p[1]
        p[0].append(p[3])
        newsymtabnode = SymtabNode('CARG', p[3], False)
        updateSymtab(newsymtabnode)

    # gatedecl
    def p_statement_gatedecl_nolr(self, p):
        """
        statement : GATE id gate_scope qarg_list gate_body 
        """
        newsymtabnode = SymtabNode('GATE', p[2]).fill_gate(p[4], p[5])
        updateSymtab(newsymtabnode)

    def p_statement_gatedecl_noargs(self, p):
        """
        statement : GATE id gate_scope '(' ')' qarg_list gate_body 
        """
        newsymtabnode = SymtabNode('GATE', p[2]).fill_gate(p[6], p[7])
        updateSymtab(newsymtabnode)

    def p_statement_gatedecl_args(self, p):
        """
        statement : GATE id gate_scope '(' carg_list ')' qarg_list gate_body 
        """
        newsymtabnode = SymtabNode('GATE', p[2]).fill_gate(p[7], p[8],p[5])
        updateSymtab(newsymtabnode)

    def p_gate_scope(self, _):
        """
        gate_scope : 
        """
        global symtab
        symtab = {}

    # gatebody
    def p_gate_body_emptybody(self, p):
        """
        gate_body : '{' gate_scope '}'
                    | '{' gate_scope error
        """
        if p[3] != '}':
            raise ParserError("Missing '}' at the end of gate definition; received " + p[3].value)
        p[0] = None

    def p_gate_body(self, p):
        """
        gate_body : '{' gop_list gate_scope '}'
                    | '{' gop_list gate_scope error
        """
        if p[4] != '}':
            raise ParserError("Missing '}' at the end of gate definition; received " + p[4].value)
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
        if len(p) == 4 and p[2] == '(':
            raise ParserError(f"Missing ')' for gate {p[1].name} at line {p[1].lineno} file {p[1].filename}")
        if len(p) == 4 and p[3] !=';':
            raise ParserError(f"Missing ';' after gate {p[1].name} at line {p[1].lineno} file {p[1].filename}")
        if len(p) == 6 and p[5] != ';':
            raise ParserError(f"Missing ';' after gate {p[1].name} at line {p[1].lineno} file {p[1].filename}")
        if len(p) == 5:
            raise ParserError(f"Invalid qubit list for gate {p[1].name} at line {p[1].lineno} file {p[1].filename}")
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
        if len(p) == 7 and p[6] != '!':
            raise ParserError(f"Missing ';' after gate {p[1].name} at line {p[1].lineno}")
        if len(p) == 6:
            raise ParserError(f"Missing qubit id after gate {p[1].name} at line {p[1].lineno}")
        if len(p) == 5:
            raise ParserError(f"Missing ')' after gate {p[1].name} at line {p[1].lineno}")
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
            raise ParserError(f"Invalid barrier qubit list inside gate definition, at line {p[1].lineno} file {p[1].filename}")
        if len(p) == 4 and p[3] != ';':
            raise ParserError(f"Missing ';' after barrier at line {p[1].lineno} file {p[1].filename}")
        p[0] = GateInstruction(node=p[1], qargs=p[2])
        self.check_gate_qargs(p[0])

    # regdecl
    def p_statement_bitdecl(self, p):
        """
        statement : qdecl ';'
                    | cdecl ';'
                    | qdecl error
                    | cdecl error
        """
        if p[2] != ';':
            raise ParserError(f"Missing ';' in qreg or creg declaration at line {p.lineno(2)}")
        p[0] = p[1]


    def p_qdecl(self, p):
        """
        qdecl : QREG indexed_id
                | QREG error
        """
        if not isinstance(p[2], IndexedId):
            raise ParserError(f"Expecting ID[int] after QREG at line {p[1].lineno} file {p[1].filename}, received {p[2].value}")
        if p[2].index <= 0:
            raise ParserError(f"QREG size must be positive at line {p[2].lineno} file {p[2].filename}")
        newsymtabnode = SymtabNode('QREG', p[2])
        updateSymtab(newsymtabnode)
        p[0] = None

    def p_qdecl(self, p):
        """
        cdecl : CREG indexed_id
                | CREG error
        """
        if not isinstance(p[2], IndexedId):
            raise ParserError(f"Expecting ID[int] after CREG at line {p[1].lineno} file {p[1].filename}, received {p[2].value}")
        if p[2].index <= 0:
            raise ParserError(f"CREG size must be positive at line {p[2].lineno} file {p[2].filename}")
        newsymtabnode = SymtabNode('CREG', p[2])
        updateSymtab(newsymtabnode)
        p[0] = None

    # id  
    def p_id(self, p):
        """
        id : ID
            | error
        """
        # It's instance of Id class, passed from t.value
        if not isinstance(p[1], Id):
            raise ParserError(f"Expected an ID, received {str(p[1].value)}")
        p[0] = p[1]


    # indexed_id
    def p_indexed_id(self, p):
        """
        indexed_id : id '[' INT ']'
                    | id '[' INT error
                    | id '[' error
        """
        if len(p) == 4 or (len(p) == 5 and p[4] != ']'):
            raise ParserError(f"Invalid character after [, received{str(p[len(p)-1].value)}")
        if len(p) == 5 and p[4] == ']':
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


    #unary
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
        
    
    #expr
    def p_expr_binary(self, p):
        """
        expression : expression '*' expression
                    | expression '/' expression
                    | expression '+' expression
                    | expression '-' expression
                    | expression '^' expression
        """
        if p[2] == '/' and p[3] == 0:
            raise ParserError(f"Divided by 0 at line {self.lexer.lineno} column {self.cal_column(self.lexer.data, p[3])}")
        if isinstance(p[1], Node) or isinstance(p[3], Node):
            p[0] = BinaryExpr(p[2], p[1], p[3])
        else:
            # int or float
            if p[2] == '*':
                p[0] = p[1] * p[3]
            elif p[2] == '/':
                p[0] = p[1] / p[3]
            elif p[2] == '^':
                p[0] = p[1] ** p[3]
            elif p[2] == '+':
                p[0] = p[1] + p[3]
            elif p[2] == '-':
                p[0] = p[1] - p[3]

    def p_expr_uminus(self, p):
        """
        expression : - expression %prec UMINUS
        """
        if isinstance(p[2], Node):
            p[0] = UnaryExpr('-', p[2])
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
            raise ParserError(f"Math function {p[1].name} not supported, only support {unaryop} line {p[1].lineno} file {p[1].filename}")
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
        
    def p_error(self, p):
        # EOF case
        if p is None:
            raise ParserError("Error at end of file")
        
        print(f"Error near line{self.lexer.lexer.lineno}, Column {self.cal_column(self.lexer.data, p)}")
    