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

"""Quafu Hamiltonian class"""

from __future__ import annotations

from collections.abc import Iterable
import numpy as np
from quafu.exceptions import QuafuError


PAULI_MAT = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex)
}


class Hamiltonian:
    """TODO"""

    def __init__(self, pauli_str_list: list[str], coeffs: np.ndarray) -> None:
        """
        Args:
            pauli_str_list: List of Pauli strs, e.g., ['IIIZZ', "IIZIZ", ...]
            coeffs: List of efficients
        """
        self._pauli_str_list = pauli_str_list
        self._coeffs = coeffs

    @staticmethod
    def from_pauli_list(pauli_list: Iterable[tuple[str, complex]]) -> Hamiltonian:
        """
        Args:
            pauli: The supported format of pauli list is [(<pauli-str>, <coefficient>)],
                e.g., [('IIIZZ', 1), ("IIZIZ", 1), ...)], 0th qubit is farthest right
        """

        pauli_list = list(pauli_list)

        size = len(pauli_list)
        if size == 0:
            raise QuafuError("Pauli list cannot be empty.")

        num_qubits = len(pauli_list[0][0])
        coeffs = np.zeros(size, dtype=complex)

        pauli_str_list = []
        for i, (pauli_str, coef) in enumerate(pauli_list):
            pauli_str_list.append(pauli_str)
            coeffs[i] = coef

        return Hamiltonian(pauli_str_list, coeffs)

    def _get_pauli_mat(self, pauli_str: str):
        """Calculate the matrix of a pauli string"""
        mat = None
        for p in pauli_str[::-1]:
            mat = PAULI_MAT[p] if mat is None else np.kron(PAULI_MAT[p], mat)
        return mat

    def matrix_generator(self):
        """Generating matrix for each Pauli str"""
        for i, p in enumerate(self._pauli_str_list):
            yield self._coeffs[i] * self._get_pauli_mat(p)

    def get_matrix(self):
        """Generate matrix of Hamiltonian"""

        mat = None
        for m in self.matrix_generator():
            if mat is None:
                mat = m
                continue
            mat += m
        return mat
