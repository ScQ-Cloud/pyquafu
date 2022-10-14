
"""
Exceptions for errors raised while building circuit.
"""

class QuafuError(Exception):
    """Base class for errors raised by Quafu."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(" ".join(message))
        self.message = " ".join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)

class CircuitError(QuafuError):
    """Exceptions for errors raised while building circuit."""
    pass

class ServerError(QuafuError):
    pass

class CompileError(QuafuError):
    pass

