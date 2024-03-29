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
from quafu.algorithms.hamiltonian import PauliOp
import quafu.elements.element_gates as qeg


def single_qubit_evol(pauli: PauliOp, time: float):
    """
    Args:
        pauli: Pauli string (little endian convention)
        time: Evolution time
    """
    gates = []

    if pauli.paulistr == "X":
        gates.append(qeg.RXGate(pauli.pos[0], 2 * time))
        return gates
    if pauli.paulistr == "Y":
        gates.append(qeg.RYGate(pauli.pos[0], 2 * time))
        return gates
    if pauli.paulistr == "Z":
        gates.append(qeg.RZGate(pauli.pos[0], 2 * time))
        return gates
    raise NotImplementedError("Unsupported Pauli string, should be in [X, Y, Z]")


def two_qubit_evol(pauli: PauliOp, time: float, cx_structure: str = "chain"):
    """
    Args:
        pauli: Pauli string (little endian convention)
        time: Evolution time
        cx_structure: Determine the structure of CX gates, can be either "chain" for
            next-neighbor connections or "fountain" to connect directly to the top qubit.
    """
    gates = []

    if pauli.paulistr == "XX":
        gates.append(qeg.RXXGate(pauli.pos[0], pauli.pos[1], 2 * time))
    elif pauli.paulistr == "YY":
        gates.append(qeg.RYYGate(pauli.pos[0], pauli.pos[1], 2 * time))
    elif pauli.paulistr == "ZZ":
        gates.append(qeg.RZZGate(pauli.pos[0], pauli.pos[1], 2 * time))
    else:
        return multi_qubit_evol(pauli, time, cx_structure)
    return gates


def multi_qubit_evol(pauli: PauliOp, time: float, cx_structure: str = "chain"):
    # determine whether the Pauli string consists of Pauli operators
    if not all(pauli_char in "XYZI" for pauli_char in pauli.paulistr):
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
    sorted_pos = sorted(pauli.pos)
    target = sorted_pos[0]

    # build the evolution as: diagonalization, reduction, 1q evolution, followed by inverses
    gates.extend(cliff)
    gates.extend(chain)
    gates.append(qeg.RZGate(target, 2 * time))
    gates.extend(chain_inverse)
    gates.extend(cliff_inverse)

    return gates


def diagonalizing_clifford(pauli: PauliOp):
    """Get the clifford gate list to diagonalize the Pauli operator.

    Args:
        pauli: The Pauli to diagonalize.

    Returns:
        A gate list for clifford.
    """
    gates = []
    gates_inverse = []

    for i, pos in enumerate(pauli.pos):
        if pauli.paulistr[i] == "Y":
            gates.append(qeg.SdgGate(pos))
            gates_inverse.append(qeg.SGate(pos))
        if pauli.paulistr[i] in ["X", "Y"]:
            gates.append(qeg.HGate(pos))
            gates_inverse.append(qeg.HGate(pos))
    gates_inverse = gates_inverse[::-1]
    return gates, gates_inverse


def cnot_chain(pauli: PauliOp):
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
    for pos in sorted(pauli.pos):
        if control is None:
            control = pos
        else:
            target = pos

        if control is not None and target is not None:
            gates.append(qeg.CXGate(control, target))
            control = pos
            target = None

    return gates, gates[::-1]


def cnot_fountain(pauli: PauliOp):
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

    for pos in sorted(pauli.pos):
        if target is None:
            target = pos
        else:
            control = pos

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

    def evol(self, pauli: PauliOp, time: float):
        num_non_id = len(pauli.paulistr)

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
