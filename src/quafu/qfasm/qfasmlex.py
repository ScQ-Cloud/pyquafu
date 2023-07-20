# Qfasm is modified OPENQASM 2.0. Currently it is mainly used to 
# transfer circuit data to backends. Further it will 
# support more instructions (classical or quantum) to enable 
# interaction with quantum hardware

import ply.lex as lex
import numpy as np


class QfasmLexer(object):

    def __init__(self):
        self.build()

    def input(self, data):
        self.data = data
        self.lexer.input(data)

    def token(self):
        ret = self.lexer.token()
        return ret

    literals = r'=()[]{};<>,.+-/*^"'

    reserved = {
        "creg": "CREG",
        "qreg": "QREG",
        "pi": "PI",
        "measure": "MEASURE",
        "include": "INCLUDE"
    }

    tokens = [
                 "FLOAT",
                 "INT",
                 "STRING",
                 "ASSIGN",
                 "MATCHES",
                 "ID",
                 "UNIT",
                 "CHANNEL",
                 "OPENQASM",
                 "NONE"
             ] + list(reserved.values())

    def t_FLOAT(self, t):
        r"(([1-9]\d*\.\d*)|(0\.\d*[1-9]\d*))"
        t.value = float(t.value)
        return t

    def t_INT(self, t):
        r"\d+"
        t.value = int(t.value)
        return t

    def t_STRING(self, t):
        r"\"([^\\\"]|\\.)*\""
        return t

    def t_ASSIGN(self, t):
        r"->"
        return t

    def t_MATCHES(self, t):
        r"=="
        return t

    def t_UNIT(self, t):
        r"ns|us"
        return t

    def t_CHANNEL(self, t):
        r"XY|Z"
        return t

    def t_OPENQASM(self, t):
        r"OPENQASM"
        return t

    def t_NONE(self, t):
        r"None"
        return t

    def t_ID(self, t):
        r"[a-z][a-zA-Z0-9_]*"
        t.type = self.reserved.get(t.value, "ID")
        return t

    t_ignore = " \t\r"

    def t_error(self, t):
        print("Illegal character '%s'" % t.value[0])

    def t_newline(self, t):
        r"\n+"
        t.lexer.lineno += len(t.value)

    def build(self, **kwargs):
        self.lexer = lex.lex(module=self, **kwargs)

    def test(self, data):
        self.lexer.input(data)
        while True:
            tok = self.lexer.token()
            if not tok:
                break
            print(tok)


if __name__ == "__main__":
    m = QfasmLexer()
    m.build()
    m.test("rx(21ns, pi) q[0], q[1]")
