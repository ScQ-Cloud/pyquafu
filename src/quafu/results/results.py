
import numpy as np
from functools import reduce
import copy
import matplotlib.pyplot as plt
from collections import OrderedDict
from ..utils.basis import *
import qutip

class Result(object):
    """Basis class for quantum results"""
    pass  

class ExecResult(Result):
    """ 
    Class that save the execute results returned from backend.
    Attributes:
        counts (dict): Samples counts on each bitstring.
        amplitudes (dict): Calculated amplitudes on each bitstring.
        taskid (int): Unique task id for the execute result.
        transpiled_circuit (QuantumCircuit): Quantum circuit transpiled on backend.
    """
    def __init__(self, input_dict, measures):
        self.measures = measures
        self.res = eval(input_dict['res'])
        self.counts = OrderedDict(sorted(self.res.items(), key=lambda s: s[0]))
        self.logicalq_res = {}
        cbits = list(self.measures.values())
        for key, values in self.counts.items():
           newkey = "".join([key[i] for i in cbits])
           self.logicalq_res[newkey] = values

        self.taskid = input_dict['task_id'] 
        self.transpiled_openqasm = input_dict["openqasm"]
        from ..circuits.quantum_circuit import QuantumCircuit
        self.transpiled_circuit = QuantumCircuit(0)
        self.transpiled_circuit.from_openqasm(self.transpiled_openqasm)
        self.measure_base = []
        total_counts = sum(self.counts.values())
        self.amplitudes = {} 
        for key in self.counts:
            self.amplitudes[key] = self.counts[key]/total_counts
    

    def calculate_obs(self, pos):
        """
        Calculate observables on input position
        Args: 
            pos (list[int]): Positions of observalbes.
        """
        return measure_obs(pos, self.logicalq_res) 

    def plot_amplitudes(self):
        """
        Plot the amplitudes from execute results.
        """
        bitstrs = list(self.amplitudes.keys())
        amps = list(self.amplitudes.values())
        plt.figure()
        plt.bar(range(len(amps)), amps, tick_label = bitstrs)
        plt.xticks(rotation=70)
        plt.ylabel("amplitudes")


class SimuResult(Result):
    """
    Class that save the execute simulation results returned from classical simulator.
    Attributes:
        num (int) : Numbers of measured qubits
        amplitudes (ndarray): Calculated amplitudes on each bitstring.
        rho (ndarray): Simulated density matrix of measured qubits
    """
    def __init__(self, input_mat, density_matrix=False):
        self.num = int(np.log2(input_mat.shape[0]))
        if density_matrix:
            self.rho = np.array(input_mat)
        else:
            self.amplitudes = input_mat
        
    def plot_amplitudes(self, full=True):
        """
        Plot the amplitudes from simulated results.
        Args:
            full (bool) : Whether plot on the full basis of measured qubits. 
        """
        from ..utils.basis import get_basis
        probs = self.amplitudes
        inds = range(len(probs))
        if not full:
            inds = np.where(self.amplitudes > 1e-14)[0]
            probs = self.amplitudes[inds]

        plt.figure()
        plt.bar(inds, probs, tick_label=[bin(i)[2:].zfill(self.num) for i in inds])
        plt.xticks(rotation=70)
        plt.ylabel("amplitudes")

    # def plot_rho(self):
    #     pass
    
    # def calculate_obs(self, obs):
    #     pass


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
            targ_basis.append(len(measure_basis)-1)
        else:
            added = 0
            for mi in range(len(measure_basis)):
                measure_base = measure_basis[mi]
                interset, intobsi, intbasei = intersec(obs[1], measure_base[1]) 
                diffset, diffobsi = diff(obs[1], measure_base[1])
                if not len(interset) == 0:
                    if all(np.array(list(obs[0]))[intobsi] == np.array(list(measure_base[0]))[intbasei]):
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
                targ_basis.append(len(measure_basis)-1)

    return measure_basis, targ_basis