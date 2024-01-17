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

from typing import Iterable
import scipy.sparse as sp
import numpy as np

from quafu.exceptions.quafu_error import QuafuError

IMat = sp.coo_matrix(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex))

XMat = sp.coo_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex))

YMat = sp.coo_matrix(np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex))

ZMat = sp.coo_matrix(np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex))

PauliMats = {"X": XMat, "Y": YMat, "Z": ZMat, "I": IMat}


class PauliOp:
    def __init__(self, paulis: str, coeff: complex = 1.0):
        paulist = paulis.split(" ")
        self.paulistr = ""
        self.pos = []
        for p in paulist:
            assert p[0] in "XYZ"
            self.paulistr += p[0]
            self.pos.append(int(p[1:]))
        self.coeff = coeff

    def __repr__(self):
        repstr = ""
        if self.coeff != 1.0:
            repstr = str(self.coeff) + "*"
        for i in range(len(self.pos)):
            repstr += self.paulistr[i]
            repstr += str(self.pos[i])
            repstr += "*"
        return repstr[:-1]

    def __str__(self):
        return self.__repr__()

    def __mul__(self, obj):
        pass

    def __rmul__(self, obj):
        pass

    def commutator(self, obj):
        pass

    def get_matrix(self, qnum, big_endian=False):
        if qnum - 1 < max(self.pos):
            raise ValueError("The support of the paulis exceed the total qubit number")

        pos = np.array(self.pos)
        if not big_endian:
            pos = qnum - 1 - pos
        inds = np.argsort(pos)
        iq = 0
        ip = 0
        mat = 1.0
        while iq < qnum:
            if ip < len(pos):
                if iq == pos[inds[ip]]:
                    opstr = self.paulistr[inds[ip]]
                    mat = sp.kron(mat, PauliMats[opstr])
                    iq += 1
                    ip += 1
                else:
                    mat = sp.kron(mat, PauliMats["I"])
                    iq += 1
            else:
                mat = sp.kron(mat, PauliMats["I"])
                iq += 1

        return self.coeff * mat


class Hamiltonian:
    def __init__(self, paulis: list[PauliOp]):
        self.paulis = paulis

    @staticmethod
    def from_pauli_list(pauli_list: Iterable[tuple[str, complex]]):
        """Generate Hamiltonian from Pauli string list

        Args:
            pauli_list: e.g., [("Z0 Z1", 1), ("Z0 Z2", 1), ...]
        """
        pauli_list = list(pauli_list)

        size = len(pauli_list)
        if size == 0:
            raise QuafuError("Pauli list cannot be empty.")

        pauli_op_list = [PauliOp(p[0], coeff=p[1]) for p in pauli_list]

        return Hamiltonian(pauli_op_list)

    def __repr__(self):
        return "+".join([str(pauli) for pauli in self.paulis])

    def __str__(self):
        return self.__repr__()

    def get_matrix(self, qnum, big_endian=False):
        mat = 0.0
        for pauli in self.paulis:
            mat += pauli.get_matrix(qnum, big_endian)

        return mat

    # TODO(zhaoyilun): delete this in the future
    def to_legacy_quafu_pauli_list(self):
        """Transform to legacy quafu pauli list format,
        this is a temperal function and should be deleted later"""
        res = []
        for pauli_str in self.paulis:
            for i, pos in enumerate(pauli_str.pos):
                res.append([pauli_str.paulistr[i], [pos]])
        return res


def intersec(a, b):
    inter = []
    aind = []
    bind = []
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i] == b[j]:
                inter.append(a[i])
                aind.append(i)
                bind.append(j)

    return inter, aind, bind


def diff(a, b):
    diff = []
    aind = []
    for i in range(len(a)):
        if a[i] not in b:
            diff.append(a[i])
            aind.append(i)

    return diff, aind


def merge_paulis(obslist):
    measure_basis = []
    targ_basis = []
    for obs in obslist:
        if len(measure_basis) == 0:
            measure_basis.append(obs)
            targ_basis.append(len(measure_basis) - 1)
        else:
            added = 0
            for mi in range(len(measure_basis)):
                measure_base = measure_basis[mi]
                interset, intobsi, intbasei = intersec(obs.pos, measure_base.pos)
                diffset, diffobsi = diff(obs.pos, measure_base.pos)
                if not len(interset) == 0:
                    if all(
                        np.array(list(obs.paulistr))[intobsi]
                        == np.array(list(measure_base.paulistr))[intbasei]
                    ):
                        measure_base.paulistr += "".join(
                            np.array(list(obs.paulistr))[diffobsi]
                        )
                        measure_base.pos.extend(diffset)
                        targ_basis.append(mi)
                        added = 1
                        break
                else:
                    measure_base.paulistr += obs.paulistr
                    measure_base.pos.extend(obs.pos)
                    targ_basis.append(mi)
                    added = 1
                    break

            if not added:
                measure_basis.append(obs)
                targ_basis.append(len(measure_basis) - 1)

    return measure_basis, targ_basis
