import numpy as np

def get_basis(ind, N):
    basisstr = bin(int(ind))[2:]
    basisstr = basisstr.zfill(N)
    basis =  [int(i) for i in basisstr] 
    return np.array(basis)

def get_ind(basis):
    biconv = 2**np.arange(len(basis))
    ind = np.dot(basis, biconv[::-1].T)
    return int(ind)
    
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

def get_baselocal(n):
    NA = n
    basisN = int(2**NA)
    baseobs = np.zeros(basisN)
    for i in range(basisN):
        basisA = get_basis(i, NA)
        baseobs[i] = (-1)**(np.sum(basisA))

    return baseobs