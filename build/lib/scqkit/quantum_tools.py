
import numpy as np
from functools import reduce
import copy
import matplotlib.pyplot as plt

class ExecResult(object):
    def __init__(self, dict, measures):
        # self.measures = dict['measure']
        self.measures = measures
        self.res = dict['res']
        self.raw_res = dict["raw"]
        self.logicalq_res = {}
        cbits = list(self.measures.values())
        for key, values in self.res.items():
           newkey = "".join([key[i] for i in np.argsort(cbits)])
           self.logicalq_res[newkey] = values

        self.taskid = dict['task_id']
        self.transpiled_openqasm = dict["openqasm"]
        from .quantum_circuit import QuantumCircuit
        self.transpiled_circuit = QuantumCircuit(0)
        self.transpiled_circuit.from_openqasm(self.transpiled_openqasm)
        self.measure_base = []       
        self.amplitudes = self.get_amplitudes()  

    def get_amplitudes(self):      
        total_counts = sum(self.res.values())
        amplitudes = {} 
        for key in self.res:
            amplitudes[key] = self.res[key]/total_counts

        return amplitudes

    def calculate_obs(self, pos):
        # return measure_obs(self.basis, self.amplitudes, pos)
        return measure_obs(pos, self.logicalq_res) #measure using frequency

    def plot_amplitudes(self):
        bitstrs = list(self.amplitudes.keys())
        amps = list(self.amplitudes.values())
        plt.figure()
        plt.bar(range(len(amps)), amps, tick_label = bitstrs)
        plt.xticks(rotation=70)
        plt.ylabel("amplitudes")
         
def reduce_probs(bitsA, res):
    """The reduced probabilities from frequency """
    dim = 2**(len(bitsA))
    probs = np.zeros(dim)
    for basestr in res:
        basis = np.array([int(i) for i in basestr])
        ind = get_ind(basis[bitsA])
        probs[ind] += res[basestr]
    
    probs = probs/np.sum(probs)
    return probs

def measure_obs(bits, res):
    n = len(bits)
    baseobs = get_baselocal(n)
    prob_r = reduce_probs(bits, res)
    result = np.dot(prob_r, baseobs)
    return result

def get_basis(ind, N):
    basisstr = bin(int(ind))[2:]
    basisstr = basisstr.zfill(N)
    basis =  [int(i) for i in basisstr] 
    return np.array(basis)

def get_ind(basis):
    biconv = 2**np.arange(len(basis))
    ind = np.dot(basis, biconv[::-1].T)
    return int(ind)

def get_baselocal(n):
    NA = n
    basisN = int(2**NA)
    baseobs = np.zeros(basisN)
    for i in range(basisN):
        basisA = get_basis(i, NA)
        baseobs[i] = (-1)**(np.sum(basisA))

    return baseobs

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