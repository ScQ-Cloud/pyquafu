from .circuits.quantum_circuit import QuantumCircuit
from .results.results import ExecResult, SimuResult 
from .tasks.tasks import Task
from .users.userapi import User
from .simulators.simulator import simulate

__all__ = ["QuantumCircuit", "ExecResult", "Task", "User", "SimuResult", "simulate"]

def get_version():
    print ("version: 0.2.9\n")
    return "0.2.9"