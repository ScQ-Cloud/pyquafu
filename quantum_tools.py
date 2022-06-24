
import numpy as np
from functools import reduce
import copy

class ExecResult(object):
    def __init__(self,dict):
        self.measures = dict['measure']
        self.res = dict['res']
        self.taskid = dict['task_id']
        # self.measure_base = dict['measure_base']        
        self.basis, self.amplitudes = self.base_ampl()  

    def base_ampl(self):      
        reslist = []
        basislist = []

        for key in self.res:
            reslist.append(self.res[key])
            basislist.append(key)
        return basislist, reslist

    def calculate_obs(self, pos):
        return measure_obs(self.basis, self.amplitudes, pos)
        # return measure_obs1(pos, self.res) #measure using frequency


def reduce_probs1(bitsA, res):
    """The reduced probabilities from frequency """
    dim = 2**(len(bitsA))
    probs = np.zeros(dim)
    for basestr in res:
        basis = np.array([int(i) for i in basestr])
        ind = get_ind(basis)
        probs[ind] += res[basestr]
    
    probs = probs/np.sum(probs)
    return probs

def measure_obs1(bits, res):
    n = len(bits)
    baseobs = get_baselocal(n)
    prob_r = reduce_probs1(bits, res)
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


def reduce_prob(basis, probs, bitA):
    lenrow = 2**(len(bitA))
    lencol = len(probs)/lenrow

    mat = np.zeros((int(lenrow), int(lencol)))
    for j in range(len(basis)):
        base = np.array([int(i) for i in basis[j]])
        baseA = base[bitA]
        bitB = [i for i in range(len(base)) if i not in bitA]
        baseB = base[bitB]
        indA = get_ind(baseA)
        indB = get_ind(baseB)
        mat[indA, indB] = probs[j]
    
    return np.sum(mat, axis=1)


def measure_obs(basis, probs, bits):
    n = len(bits)
    baseobs = get_baselocal(n)
    prob_r = reduce_prob(basis, probs, bits)
    result = np.dot(prob_r, baseobs)
    return result

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