from .circuits.quantum_circuit import QuantumCircuit
from .circuits.quantum_register import QuantumRegister, Qubit
from .results.results import ExecResult, SimuResult
from .tasks.tasks import Task
from .users.userapi import User
from .simulators.simulator import simulate

__all__ = [
    "QuantumCircuit",
    "QuantumRegister",
    "Qubit",
    "ExecResult",
    "Task",
    "User",
    "SimuResult",
    "simulate",
    "get_version",
]


def get_version():
    return "0.4.0"
