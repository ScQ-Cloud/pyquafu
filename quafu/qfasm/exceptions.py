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
