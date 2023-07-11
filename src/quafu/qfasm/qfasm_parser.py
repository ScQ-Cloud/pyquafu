import ply.yacc as yacc

from quafu.dagcircuits.instruction_node import InstructionNode
from .qfasmlex import QfasmLexer


import numpy as np


class DeclarationNode(object):
    pass


class QregNode(DeclarationNode):
    def __init__(self, n):
        self.n = n

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "qreg[%d]" % self.n


class CregNode(DeclarationNode):
    def __init__(self, n):
        self.n = n

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "creg[%d]" % self.n


class IncludeNode(DeclarationNode):
    def __init__(self, filename):
        self.file = filename

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "include %s" % self.file


class OPENQASMNode(DeclarationNode):
    def __init__(self, version):
        self.version = version

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "OPENQASM %.1f" % self.version


class QfasmParser(object):
    tokens = QfasmLexer.tokens

    def __init__(self, debug=False):
        self.parser = yacc.yacc(module=self, debug=debug)
        self.parsed_nodes = []
        self.lexer = QfasmLexer()

    def parse(self, input: str):
        self.parsed_nodes = self.parser.parse(input, lexer=QfasmLexer())
        return self.parsed_nodes

    def p_main_0(self, p):
        """
        main : program
        """
        p[0] = [p[1]]

    def p_main_1(self, p):
        """
        main : main program
        """
        p[1].append(p[2])
        p[0] = p[1]

    def p_program(self, p):
        """
        program : instruction
                | declaration
        """
        p[0] = p[1]

    def p_declaration(self, p):
        """
        declaration : openqasm
                    | include
                    | qreg
                    | creg
        """
        p[0] = p[1]

    def p_openqasm(self, p):
        """
        openqasm : OPENQASM FLOAT ';'
        """
        p[0] = OPENQASMNode(p[2])

    def p_include(self, p):
        """
        include : INCLUDE STRING ';'
        """
        p[0] = IncludeNode(p[2])

    def p_qreg(self, p):  # TODO:verify register name
        """
        qreg : QREG bitreg ';'
        """
        p[0] = QregNode(p[2])

    def p_creg(self, p):
        """
        creg : CREG bitreg ';'
        """
        p[0] = CregNode(p[2])

    def p_instruction(self, p):
        """
        instruction : gate ';'
                | pulse ';'
                | measure ';'
        """
        p[0] = p[1]

    def p_arg_list_0(self, p):
        """
        arg_list : expression
        """
        p[0] = [p[1]]

    def p_arg_list_1(self, p):
        """
        arg_list : arg_list ',' expression
        """
        p[1].append(p[3])
        p[0] = p[1]

    def p_gate_like_0(self, p):
        '''
        gate : id qubit_list
        '''
        p[0] = InstructionNode(p[1], p[2], None, None, None, "", None, "")

    def p_gate_like_1(self, p):
        '''
        gate : id '(' arg_list ')' qubit_list
        '''
        p[0] = InstructionNode(p[1], p[5], p[3], None, None, "", None, "")

    def p_pulse_like_0(self, p):
        '''
        pulse : id '(' time ',' arg_list ')' qubit_list
        '''
        p[0] = InstructionNode(p[1], p[7], p[5], p[3][0], p[3][1], "", None, "")

    def p_measure_0(self, p):
        '''
        measure : MEASURE bitreg ASSIGN bitreg
        '''
        p[0] = InstructionNode("measure", {p[2]: p[4]}, None, None, None, "", None, "")

    def p_pulse_like_1(self, p):
        '''
        pulse : id '(' time ',' arg_list ',' channel ')' qubit_list
        '''
        p[0] = InstructionNode(p[1], p[9], p[5], p[3][0], p[3][1], p[7], None, "")

    def p_bitreg(self, p):
        """
        bitreg : id '[' INT ']'
        """
        p[0] = p[3]

    def p_qubit_list_0(self, p):
        """
        qubit_list : bitreg
        """
        p[0] = [p[1]]

    def p_qubit_list_1(self, p):
        """
        qubit_list : qubit_list ',' bitreg
        """
        p[1].append(p[3])
        p[0] = p[1]

    def p_channel(self, p):
        """
        channel : CHANNEL
        """
        p[0] = p[1]

    def p_time(self, p):
        """
        time : INT UNIT
        """
        p[0] = (p[1], p[2])

    def p_expression_none(self, p):
        """
        expression : NONE
        """
        p[0] = p[1]

    def p_expression_term(self, p):
        'expression : term'
        p[0] = p[1]

    def p_expression_m(self, p):
        """
        expression : '-' expression
        """
        p[0] = -p[2]

    def p_term_factor(self, p):
        """
        term : factor
        """
        p[0] = p[1]

    def p_binary_operators(self, p):
        '''expression : expression '+' term
                    | expression '-' term
        term       : term '*' factor
                    | term '/' factor'''
        if p[2] == '+':
            p[0] = p[1] + p[3]
        elif p[2] == '-':
            p[0] = p[1] - p[3]
        elif p[2] == '*':
            p[0] = p[1] * p[3]
        elif p[2] == '/':
            p[0] = p[1] / p[3]

    def p_factor_0(self, p):
        '''
        factor : FLOAT
            | INT
        '''
        p[0] = p[1]

    def p_factor_1(self, p):
        '''
        factor : '(' expression ')'
        '''
        p[0] = p[2]

    def p_factor_pi(self, p):
        '''
        factor : PI
        '''
        p[0] = np.pi

    def p_id(self, p):
        ''' id : ID'''
        p[0] = p[1]

    def p_error(self, p):
        if p:
            print("Syntax error at token", p.type)
            # Just discard the token and tell the parser it's okay.
            self.parser.errok()
        else:
            print("Syntax error at EOF")
