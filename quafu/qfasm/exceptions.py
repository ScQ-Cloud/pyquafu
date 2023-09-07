<<<<<<< HEAD:src/quafu/qfasm/exceptions.py
class LexerError(Exception):
    """Errors raised while lexer get tokens"""
    pass

class ParserError(Exception):
    """Errors raised while parser OPENQASM"""
    pass
=======
from quafu.exceptions import QuafuError


class QfasmError(QuafuError):
    """
    Base class for errors raised by Qfasm.
    """
    pass


class QfasmSyntaxError(QfasmError):
    pass


class QfasmSemanticError(QfasmError):
    pass
>>>>>>> master:quafu/qfasm/exceptions.py
