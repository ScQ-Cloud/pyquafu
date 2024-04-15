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

# Qfasm is modified OPENQASM 2.0. Currently it is mainly used to
# transfer circuit data to backends. Further it will
# support more instructions (classical or quantum) to enable
# interaction with quantum hardware

import os

import ply.lex as lex

from .exceptions import LexerError
from .qfasm_utils import Id


class QfasmLexer(object):
    def __init__(self, filename: str = None):
        self.build(filename)
        self.file_lexer_stack = []

    def build(self, filename: str = None):
        self.lexer = lex.lex(module=self)
        self.lexer.filename = filename
        self.lexer.lineno = 1
        if filename:
            with open(filename, "r") as ifile:
                self.data = ifile.read()
            self.lexer.input(self.data)

    def push_lexer(self, filename):
        self.file_lexer_stack.append(self.lexer)
        self.build(filename=filename)

    def pop_lexer(self):
        self.lexer = self.file_lexer_stack.pop()

    def input(self, data):
        self.data = data
        self.lexer.input(data)
        # read qelib1.inc
        qelib1 = os.path.join(os.path.dirname(__file__), "qelib1.inc")
        self.push_lexer(qelib1)

    def token(self):
        ret = self.lexer.token()
        return ret

    literals = r'()[]{};<>,+-/*^"'

    reserved = {
        "creg": "CREG",
        "qreg": "QREG",
        "pi": "PI",
        "measure": "MEASURE",
        "include": "INCLUDE",
        "barrier": "BARRIER",
        "gate": "GATE",
        "opaque": "OPAQUE",
        "reset": "RESET",
        "if": "IF",
    }

    tokens = [
        "FLOAT",
        "INT",
        "STRING",
        "ASSIGN",
        "MATCHES",
        "EQUAL",
        "ID",
        "UNIT",
        "CHANNEL",
        "OPENQASM",
        "NONE",
    ] + list(reserved.values())

    # dispose include file
    def t_INCLUDE(self, _):
        "include"
        filename_token = self.lexer.token()
        if filename_token is None:
            raise LexerError("Expecting filename, received nothing.")
        # print(filename_token.value)
        if isinstance(filename_token.value, str):
            filename = filename_token.value.strip('"')
            if filename == "":
                raise LexerError("Invalid include: need a quoted string as filename.")
        else:
            raise LexerError("Invalid include: need a quoted string as filename.")

        # just ignore, because we include it at first
        if filename == "qelib1.inc":
            semicolon_token = self.lexer.token()
            if semicolon_token is None or semicolon_token.value != ";":
                raise LexerError(
                    f'Expecting ";" for INCLUDE at line {semicolon_token.lineno}, in file {self.lexer.filename}'
                )
            return self.lexer.token()
        # if filename == 'qelib1.inc':
        #     filename = os.path.join(os.path.dirname(__file__), 'qelib1.inc')

        if not os.path.exists(filename):
            # print(filename)
            raise LexerError(
                f"Include file {filename} cannot be found, at line {filename_token.lineno}, in file {self.lexer.filename}"
            )

        semicolon_token = self.lexer.token()
        if semicolon_token is None or semicolon_token.value != ";":
            raise LexerError(
                f'Expecting ";" for INCLUDE at line {semicolon_token.lineno}, in file {self.lexer.filename}'
            )

        self.push_lexer(filename)
        return self.lexer.token()

    def t_FLOAT(self, t):
        r"([0-9]+\.\d*(e[-+]?[0-9]+)?)"
        t.value = float(t.value)
        return t

    def t_INT(self, t):
        r"\d+"
        t.value = int(t.value)
        return t

    def t_STRING(self, t):
        r'"([^\\\"]|\\.)*"'
        return t

    def t_ASSIGN(self, t):
        r"->"
        return t

    def t_MATCHES(self, t):
        r"=="
        return t

    def t_EQUAL(self, t):
        r"="
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

    def t_CX_U(self, t):
        "CX|U"
        t.type = "ID"
        t.value = Id(t.value, self.lexer.lineno, self.lexer.filename)
        return t

    def t_ID(self, t):
        r"[a-z][a-zA-Z0-9_]*"
        t.type = self.reserved.get(t.value, "ID")
        # all the reset | barrier | gate | include  | measure | opaque
        t.value = Id(t.value, self.lexer.lineno, self.lexer.filename)
        return t

    def t_COMMENT(self, _):
        r"//.*"
        pass

    t_ignore = " \t\r"

    def t_error(self, t):
        raise LexerError(f"Illegal character {t.value[0]}")

    def t_newline(self, t):
        r"\n+"
        t.lexer.lineno += len(t.value)

    def t_eof(self, _):
        if len(self.file_lexer_stack) > 0:
            self.pop_lexer()
            return self.lexer.token()
        return None

    def test_data(self, data):
        self.lexer.input(data)
        while True:
            tok = self.lexer.token()
            if not tok:
                break
            print(tok)

    def test_file(self):
        print_file = open("lex.txt", "w+")
        while True:
            tok = self.lexer.token()
            if not tok:
                break
            print(tok, file=print_file)
