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
"""Measurement result."""

import copy
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

from ..utils.basis import measure_obs


# pylint: disable=too-few-public-methods
class Result:
    """Basis class for quantum results"""


# pylint: disable=too-many-instance-attributes
class ExecResult(Result):
    """
    Class that save the execute results returned from backend.

    Attributes:
        counts (dict): Samples counts on each bitstring.
        probabilities (dict): Calculated probabilities on each bitstring.
        taskid (int): Unique task id for the execute result.
        transpiled_circuit (QuantumCircuit): Quantum circuit transpiled on backend.
    """

    def __init__(self, input_dict):
        status_map = {
            0: "In Queue",
            1: "Running",
            2: "Completed",
            3: "Canceled",
            4: "Failed",
            5: "Pending",
        }
        self.taskid = input_dict["task_id"]
        self.taskname = input_dict["task_name"]
        self.transpiled_openqasm = input_dict["openqasm"]
        from ..circuits.quantum_circuit import (  # pylint: disable=import-outside-toplevel
            QuantumCircuit,
        )

        self.transpiled_circuit = QuantumCircuit(0)
        self.transpiled_circuit.from_openqasm(self.transpiled_openqasm)
        self.measure_base = []

        self.measures = self.transpiled_circuit.measures
        self.task_status = status_map[input_dict["status"]]
        self.res = eval(input_dict["res"])  # noqa:SCS101  # pylint: disable=eval-used,avoid-eval-exec
        self.counts = OrderedDict(sorted(self.res.items(), key=lambda s: s[0]))

        self.logicalq_res = {}
        cbits = list(self.measures.values())
        indexed_cbits = {bit: i for i, bit in enumerate(sorted(cbits))}
        squeezed_cbits = [indexed_cbits[bit] for bit in cbits]
        for key, values in self.counts.items():
            newkey = "".join([key[i] for i in squeezed_cbits])
            self.logicalq_res[newkey] = values

        total_counts = sum(self.counts.values())
        self.probabilities = {}
        for bit_str in self.counts:
            self.probabilities[bit_str] = self.counts[bit_str] / total_counts

    def calculate_obs(self, pos):
        """
        Calculate observables Z on input position using probabilities

        Args:
            pos (list[int]): Positions of observalbes.
        """
        return measure_obs(pos, self.logicalq_res)

    def plot_probabilities(self):
        """
        Plot the probabilities from execute results.
        """
        bitstrs = list(self.probabilities.keys())
        probs = list(self.probabilities.values())
        plt.figure()
        plt.bar(range(len(probs)), probs, tick_label=bitstrs)
        plt.xticks(rotation=70)
        plt.ylabel("probabilities")


class SimuResult(Result):
    def __init__(self, res_info: dict):
        """
        Args:
            res_info: data from simulator
        """
        self._meta_data = res_info
        counts = res_info["counts"]
        if counts:
            if isinstance(list(counts.keys())[0], int):
                cbits_num = len(res_info["measures"])
                newcounts = {}
                for key, value in counts.items():
                    bitstr = bin(key)[2:].zfill(cbits_num)
                    newcounts[bitstr] = value
                self._meta_data["counts"] = newcounts

        self._probabilities = []

    def __getitem__(self, key: str):
        """
        Get meta_data of simulate results.
        Args:
            `"statevector"`: full state vector
            `"counts"`: sampled  bitstring counts
            `"pauli_expects"`: pauli expectations of input paulistrings
        """
        return self._meta_data[key]

    def get_statevector(self):
        try:
            return self["statevector"]
        except KeyError as exc:
            raise KeyError(
                f"no statevector saved from {self['simulator']} simulator"
            ) from exc

    @property
    def probabilities(self):
        if len(self._probabilities) == 0:
            self.calc_probabilities()
        return self._probabilities

    @property
    def counts(self):
        return self["counts"]

    def calc_probabilities(self, from_counts=False):
        num = self["qbitnum"]
        if from_counts and self._meta_data["counts"]:
            counts = self._meta_data["counts"]
            total_counts = sum(counts.values())
            probabilities = {}
            for key in self._meta_data["counts"]:
                probabilities[key] = counts[key] / total_counts
            self._probabilities = np.zeros(2**num)
            for key in probabilities:
                self._probabilities[int(key, 2)] = probabilities[key]

        elif "statevector" in self._meta_data.keys():
            psi = self.get_statevector()
            measures = list(self["measures"].keys())
            values_tmp = list(self["measures"].values())
            values = np.argsort(values_tmp)

            # pylint: disable=import-outside-toplevel
            from quafu.simulators.default_simulator import permutebits, ptrace

            psi = permutebits(psi, range(num)[::-1])
            if measures:
                self._probabilities = ptrace(psi, measures)
                self._probabilities = permutebits(self._probabilities, values)
            else:
                self._probabilities = np.abs(psi) ** 2
        else:
            raise ValueError("No data saved for probs")

    def plot_probabilities(
        self,
        full: bool = False,
        reverse_basis: bool = False,
        sort: bool = None,
        from_counts=False,
    ):
        """
        Plot the probabilites of measured qubits
        """

        if from_counts:
            counts = self._meta_data["counts"]
            total_counts = sum(counts.values())
            probabilities = {}
            for key in self._meta_data["counts"]:
                probabilities[key] = counts[key] / total_counts

            bitstrs = list(probabilities.keys())
            probs = list(probabilities.values())
            plt.figure()
            plt.bar(range(len(probs)), probs, tick_label=bitstrs)
            plt.xticks(rotation=70)
            plt.ylabel("probabilities")

        elif len(self.get_statevector()) > 0:
            if not full:
                inds = np.where(self.probabilities > 1e-14)[0]
                probs = self.probabilities[inds]

            measures = self._meta_data["measures"]
            num = len(measures) if measures else self["qbitnum"]
            basis = np.array([bin(i)[2:].zfill(num) for i in inds])
            if reverse_basis:
                basis = np.array([bin(i)[2:].zfill(num)[::-1] for i in inds])

            if sort == "ascend":
                orders = np.argsort(probs)
                probs = probs[orders]
                basis = basis[orders]
            elif sort == "descend":
                orders = np.argsort(probs)
                probs = probs[orders][::-1]
                basis = basis[orders][::-1]

            plt.figure()
            plt.bar(inds, probs, tick_label=basis)
            plt.xticks(rotation=70)
            plt.ylabel("probabilities")
        else:
            raise ValueError("No data for ploting")

    def calc_density_matrix(self):
        psi = self.get_statevector()
        num = self["qbitnum"]
        measures = list(self["measures"].keys())
        values_tmp = list(self["measures"].values())
        values = np.argsort(values_tmp)
        if len(measures) == 0:
            measures = list(range(num))
            values = list(range(num))
        from quafu.simulators.default_simulator import (  # pylint: disable=import-outside-toplevel
            permutebits,
            ptrace,
        )

        psi = permutebits(psi, range(num)[::-1])
        rho = ptrace(psi, measures, diag=False)
        return permutebits(rho, values)


# TODO:These should merge to paulis
def intersec(a, b):
    inter = []
    aind = []
    bind = []
    for i, a_i in enumerate(a):
        for j, b_j in enumerate(b):
            if a_i == b_j:
                inter.append(a_i)
                aind.append(i)
                bind.append(j)

    return inter, aind, bind


def diff(a, b):
    diff_list = []
    aind = []
    for i, a_i in enumerate(a):
        if a_i not in b:
            diff_list.append(a_i)
            aind.append(i)

    return diff_list, aind


def merge_measure(obslist):
    obslist = copy.deepcopy(obslist)
    measure_basis = []
    targ_basis = []
    for obs in obslist:
        if len(measure_basis) == 0:
            measure_basis.append(obs)
            targ_basis.append(len(measure_basis) - 1)
        else:
            added = 0
            for mi, measure_base in enumerate(measure_basis):
                interset, intobsi, intbasei = intersec(obs[1], measure_base[1])
                diffset, diffobsi = diff(obs[1], measure_base[1])
                if len(interset) != 0:
                    if all(
                        np.array(list(obs[0]))[intobsi]
                        == np.array(list(measure_base[0]))[intbasei]
                    ):
                        measure_base[0] += "".join(np.array(list(obs[0]))[diffobsi])
                        measure_base[1].extend(diffset)
                        targ_basis.append(mi)
                        added = 1
                        break
                else:
                    measure_base[0] += obs[0]
                    measure_base[1].extend(obs[1])
                    targ_basis.append(mi)
                    added = 1
                    break

            if not added:
                measure_basis.append(obs)
                targ_basis.append(len(measure_basis) - 1)

    return measure_basis, targ_basis
