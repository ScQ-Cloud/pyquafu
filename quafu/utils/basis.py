import numpy as np


def get_basis(ind, n):
    basisstr = bin(int(ind))[2:]
    basisstr = basisstr.zfill(n)
    basis = [int(i) for i in basisstr]
    return np.array(basis)


def get_ind(basis):
    biconv = 2 ** np.arange(len(basis))
    ind = np.dot(basis, biconv[::-1].T)
    return int(ind)


def reduce_probs(bits_a, res):
    """The reduced probabilities from frequency"""
    dim = 2 ** (len(bits_a))
    probs = np.zeros(dim)
    for basestr in res:
        basis = np.array([int(i) for i in basestr])
        ind = get_ind(basis[bits_a])
        probs[ind] += res[basestr]

    probs = probs / np.sum(probs)
    return probs


def measure_obs(bits, res):
    n = len(bits)
    baseobs = get_baselocal(n)
    prob_r = reduce_probs(bits, res)
    return np.dot(prob_r, baseobs)


def get_baselocal(n):
    na = n
    basis_n = int(2**na)
    baseobs = np.zeros(basis_n)
    for i in range(basis_n):
        basis_a = get_basis(i, na)
        baseobs[i] = (-1) ** (np.sum(basis_a))

    return baseobs
