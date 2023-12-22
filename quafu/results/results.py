import copy
from collections import OrderedDict

import matplotlib.pyplot as plt

from ..utils.basis import *


class Result(object):
    """Basis class for quantum results"""

    pass


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
            "Canceled": 3,
            4: "Failed",
        }
        self.taskid = input_dict["task_id"]
        self.taskname = input_dict["task_name"]
        self.transpiled_openqasm = input_dict["openqasm"]
        from ..circuits.quantum_circuit import QuantumCircuit

        self.transpiled_circuit = QuantumCircuit(0)
        self.transpiled_circuit.from_openqasm(self.transpiled_openqasm)
        self.measure_base = []

        self.measures = self.transpiled_circuit.measures
        self.task_status = status_map[input_dict["status"]]
        self.res = eval(input_dict["res"])
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
    """
    Class that save the execute simulation results returned from classical simulator.

    Attributes:
        num (int): Numbers of measured qubits.
        probabilities (ndarray): Calculated probabilities on each bitstring.
        rho (ndarray): Simulated density matrix of measured qubits.
        count_dict: The num of cbits measured. Only support for `qfvm_circuit`.
    """

    def __init__(self, input, input_form, count_dict: dict = None):
        if input_form != "count_dict":
            self.num = int(np.log2(input.shape[0]))
        else:
            # input is num qubits
            self.num = input
        if input_form == "density_matrix":
            self.rho = np.array(input)
            self.probabilities = np.diag(input)
        elif input_form == "probabilities":
            self.probabilities = input
        elif input_form == "state_vector":
            self.state_vector = input
        elif input_form == "count_dict":
            # do nothing, only count dict
            pass
        # come form c++ simulator
        # TODO: add count for py_simu
        if count_dict is not None:
            self.count = {}
            for key, value in count_dict.items():
                bitstr = bin(key)[2:].zfill(self.num)
                self.count[bitstr] = value

    def plot_probabilities(
        self, full: bool = False, reverse_basis: bool = False, sort: bool = None
    ):
        """
        Plot the probabilities from simulated results, ordered in big endian convention.

        Args:
            full: Whether plot on the full basis of measured qubits.
            reverse_basis: Whether reverse the bitstring of basis. (Little endian convention).
            sort:  Sort the results by probabilities values. Can be `"ascend"` order or `"descend"` order.
        """

        probs = self.probabilities
        inds = range(len(probs))
        if not full:
            inds = np.where(self.probabilities > 1e-14)[0]
            probs = self.probabilities[inds]

        basis = np.array([bin(i)[2:].zfill(self.num) for i in inds])
        if reverse_basis:
            basis = np.array([bin(i)[2:].zfill(self.num)[::-1] for i in inds])

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

    def get_statevector(self):
        return self.state_vector

    def calculate_obs(self, pos):
        "Calculate observables Z on input position using probabilities"
        inds = np.where(self.probabilities > 1e-14)[0]
        probs = self.probabilities[inds]
        basis = np.array([bin(i)[2:].zfill(self.num) for i in inds])
        res_reduced = dict(zip(basis, probs))
        return measure_obs(pos, res_reduced)


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
            for mi in range(len(measure_basis)):
                measure_base = measure_basis[mi]
                interset, intobsi, intbasei = intersec(obs[1], measure_base[1])
                diffset, diffobsi = diff(obs[1], measure_base[1])
                if not len(interset) == 0:
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
