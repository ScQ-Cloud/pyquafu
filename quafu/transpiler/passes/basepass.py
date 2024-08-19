from abc import ABC, abstractmethod
from typing import Union

from quafu import QuantumCircuit
from quafu.elements import Instruction

from quafu.dagcircuits.dag_circuit import DAGCircuit


class BasePass(ABC):
    """
    The Metaclass for quafu.transpile compiler passes.
    """

    @abstractmethod
    def run(self, circuit: Union[QuantumCircuit, DAGCircuit, Instruction]):
        pass


class UnrollPass(BasePass):
    """
    The UnrollPass for quafu.transpile compiler unroll.
    """

    def __init__(self) -> None:
        self.rule = []
        self.parameter_type = 'constant_gate'  # or 'parameterized_gate'
        self.original = 'cx'
        self.basis = ['cx', 'rx', 'ry', 'rz', 'id']
        self.global_phase = 0

    @abstractmethod
    def run(self, circuit: Union[QuantumCircuit, DAGCircuit, Instruction]):
        pass

