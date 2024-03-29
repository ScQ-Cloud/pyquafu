from .circuits.quantum_circuit import QuantumCircuit
from .algorithms.hamiltonian import Hamiltonian
from .circuits.quantum_register import QuantumRegister, Qubit
from .results.results import ExecResult, SimuResult
from .simulators import simulate
from .tasks.tasks import Task
from .users.userapi import User

__all__ = [
    "QuantumCircuit",
    "QuantumRegister",
    "Qubit",
    "Hamiltonian",
    "ExecResult",
    "Task",
    "User",
    "SimuResult",
    "simulate",
    "get_version",
]


def get_version():
    return "0.4.0"
