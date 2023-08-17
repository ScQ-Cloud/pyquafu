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

"""Evolution that generate gate sequence based on operator"""

from abc import ABC, abstractmethod
import numpy as np
import quafu.elements.element_gates as qeg


def two_qubit_evol(pauli: str, time: float, cx_structure: str = "chain"):
    """
    Args:
        pauli: Pauli string
        time: Evolution time
        cx_structure: TODO
    """
    qubits = [i for i in range(len(pauli)) if pauli[i] != "I"]
    labels = np.array([pauli[i] for i in qubits])
    gates = []

    if all(labels == "X"):
        pass
    elif all(labels == "Y"):
        pass
    elif all(labels == "Z"):
        gates.append(qeg.RZZGate(qubits[0], qubits[1], 2 * time))
    else:
        raise NotImplementedError("Pauli string not yet supported")
    return gates


class BaseEvolution(ABC):
    """Generate evolution circuit based on operators"""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def evol(self, pauli: str, time: float):
        """Generate gate sequence based on input operator

        Args:
            pauli: The pauli to evolve.
            time: Evolution time.
        """
        pass


class ProductFormula(BaseEvolution):
    """Product formula for decomposition of operator exponentials"""

    def evol(self, pauli: str, time: float):
        num_non_id = len([label for label in pauli if label != "I"])

        if num_non_id == 0:
            pass
        elif num_non_id == 1:
            pass
        elif num_non_id == 2:
            return two_qubit_evol(pauli, time)
        else:
            raise NotImplementedError(f"Pauli string {pauli} not yet supported")
        return []
