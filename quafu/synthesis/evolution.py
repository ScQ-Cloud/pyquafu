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


def single_qubit_evol(pauli: str, time: float):
    """
    Args:
        pauli: Pauli string (little endian convention)
        time: Evolution time
    """
    reversed_pauli = pauli[::-1]
    gates = []

    for i, pauli_i in enumerate(reversed_pauli):
        if pauli_i == "I":
            continue
        elif pauli_i == "X":
            gates.append(qeg.RXGate(i, 2 * time))
            return gates
        elif pauli_i == "Y":
            gates.append(qeg.RYGate(i, 2 * time))
            return gates
        elif pauli_i == "Z":
            gates.append(qeg.RZGate(i, 2 * time))
            return gates
        else:
            raise NotImplementedError("Pauli string not yet supported")


def two_qubit_evol(pauli: str, time: float, cx_structure: str = "chain"):
    """
    Args:
        pauli: Pauli string (little endian convention)
        time: Evolution time
        cx_structure: Determine the structure of CX gates, can be either "chain" for
            next-neighbor connections or "fountain" to connect directly to the top qubit.
    """
    reversed_pauli = pauli[::-1]
    qubits = [i for i in range(len(reversed_pauli)) if reversed_pauli[i] != "I"]
    labels = np.array([reversed_pauli[i] for i in qubits])
    gates = []

    if all(labels == "X"):
        gates.append(qeg.RXXGate(qubits[0], qubits[1], 2 * time))
    elif all(labels == "Y"):
        gates.append(qeg.RYYGate(qubits[0], qubits[1], 2 * time))
    elif all(labels == "Z"):
        gates.append(qeg.RZZGate(qubits[0], qubits[1], 2 * time))
    else:
        return multi_qubit_evol(pauli, time, cx_structure)
    return gates


def multi_qubit_evol(pauli: str, time: float, cx_structure: str = "chain"):
    # determine whether the Pauli string consists of Pauli operators
    if not all(pauli_char in "XYZI" for pauli_char in pauli):
        raise NotImplementedError("Pauli string not yet supported")
    gates = []
    # get diagonalizing clifford gate list
    cliff, cliff_inverse = diagonalizing_clifford(pauli)

    # get CX chain to reduce the evolution to the top qubit
    if cx_structure == "chain":
        chain, chain_inverse = cnot_chain(pauli)
    else:
        chain, chain_inverse = cnot_fountain(pauli)

    # determine qubit to do the rotation on
    target = None
    # Note that all phases are removed from the pauli label and are only in the coefficients.
    # That's because the operators we evolved have all been translated to a SparsePauliOp.
    for i, pauli_i in enumerate(pauli[::-1]):
        if pauli_i != "I":
            target = i
            break

    # build the evolution as: diagonalization, reduction, 1q evolution, followed by inverses
    gates.extend(cliff)
    gates.extend(chain)
    gates.append(qeg.RZGate(target, 2 * time))
    gates.extend(chain_inverse)
    gates.extend(cliff_inverse)

    return gates


def diagonalizing_clifford(pauli: str):
    """Get the clifford gate list to diagonalize the Pauli operator.

    Args:
        pauli: The Pauli to diagonalize.

    Returns:
        A gate list for clifford.
    """
    reversed_pauli = pauli[::-1]
    gates = []
    gates_inverse = []

    for i, pauli_i in enumerate(reversed_pauli):
        if pauli_i == "Y":
            gates.append(qeg.SdgGate(i))
            gates_inverse.append(qeg.SGate(i))
        if pauli_i in ["X", "Y"]:
            gates.append(qeg.HGate(i))
            gates_inverse.append(qeg.HGate(i))
    gates_inverse = gates_inverse[::-1]
    return gates, gates_inverse


def cnot_chain(pauli: str):
    """CX chain.

    For example, for the Pauli with the label 'XYZIX'.

                       ┌───┐
        q_0: ──────────┤ X ├
                       └─┬─┘
        q_1: ────────────┼──
                  ┌───┐  │
        q_2: ─────┤ X ├──■──
             ┌───┐└─┬─┘
        q_3: ┤ X ├──■───────
             └─┬─┘
        q_4: ──■────────────

    Args:
        pauli: The Pauli for which to construct the CX chain.

    Returns:
        A gate list implementing the CX chain.
    """

    gates = []
    control, target = None, None

    # iterate over the Pauli's and add CNOTs
    for i, pauli_i in enumerate(pauli):
        i = len(pauli) - i - 1
        if pauli_i != "I":
            if control is None:
                control = i
            else:
                target = i

        if control is not None and target is not None:
            gates.append(qeg.CXGate(control, target))
            control = i
            target = None

    return gates, gates[::-1]


def cnot_fountain(pauli: str):
    """CX chain in the fountain shape.

    For example, for the Pauli with the label 'XYZIX'.

             ┌───┐┌───┐┌───┐
        q_0: ┤ X ├┤ X ├┤ X ├
             └─┬─┘└─┬─┘└─┬─┘
        q_1: ──┼────┼────┼──
               │    │    │
        q_2: ──■────┼────┼──
                    │    │
        q_3: ───────■────┼──
                         │
        q_4: ────────────■──

    Args:
        pauli: The Pauli for which to construct the CX chain.

    Returns:
        A gate list implementing the CX chain.
    """

    gates = []
    control, target = None, None
    for i, pauli_i in enumerate(pauli[::-1]):
        if pauli_i != "I":
            if target is None:
                target = i
            else:
                control = i

        if control is not None and target is not None:
            gates.append(qeg.CXGate(control, target))
            control = None

    return gates, gates[::-1]


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
            return single_qubit_evol(pauli, time)
        elif num_non_id == 2:
            return two_qubit_evol(pauli, time)
        else:
            # raise NotImplementedError(f"Pauli string {pauli} not yet supported")
            return multi_qubit_evol(pauli, time)
        return []
