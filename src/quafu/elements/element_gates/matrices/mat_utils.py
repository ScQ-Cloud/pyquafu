import cmath

import numpy as np
from numpy import ndarray

from .mat_lib import IdMatrix


def split_matrix(matrix: ndarray):
    """
    Evenly split a matrix into 4 sub-matrices.
    """
    top, bottom = np.vsplit(matrix, 2)
    t_left, t_right = np.hsplit(top, 2)
    b_left, b_right = np.hsplit(bottom, 2)
    return t_left, t_right, b_left, b_right


def stack_matrices(t_left, t_right, b_left, b_right):
    """
    Stack 4 sub-matrices into a matrix.
    """
    top = np.hstack((t_left, t_right))
    bottom = np.hstack((b_left, b_right))
    mat = np.vstack((top, bottom))
    return mat


def multi_kron(op1, op2, ind1, ind2, nspin):
    tmp = 1
    for i in range(nspin):
        if i == ind1:
            tmp = np.kron(tmp, op1)
        elif i == ind2:
            tmp = np.kron(tmp, op2)
        else:
            tmp = np.kron(tmp, IdMatrix)
    return tmp


def general_kron(op, ind, nqubit):
    tmp = 1
    for i in range(nqubit):
        if i == ind:
            tmp = np.kron(tmp, op)
        else:
            tmp = np.kron(tmp, IdMatrix)
    return tmp


#######################################################
def is_zero(a):
    return not np.any(np.absolute(a) > 1e-8)


def is_approx(a, b, thres=1e-6):
    # TODO: seems there are some very small elements that cannot be compared correctly
    # if not np.allclose(a, b, rtol=thres, atol=thres):
    #    print(np.sum(a-b))
    return np.allclose(a, b, rtol=thres, atol=thres)


def is_unitary(matrix):
    mat_dg = np.conjugate(matrix).T
    id_mat = np.eye(matrix.shape[0])
    return is_approx(mat_dg @ matrix, id_mat) and is_approx(matrix @ mat_dg, id_mat)


def is_hermitian(matrix):
    tmp = np.conjuate(matrix).T
    return is_approx(tmp, matrix)


def is_diagonal(matrix: ndarray):
    diag = np.diag(matrix)
    diag_mat = np.diag(diag)
    return is_approx(matrix, diag_mat)


def is_kron_with_id2(matrix):
    """
    Check if the matrix is a Kronecker product of a matrix and identity matrix.
    """
    nsize = matrix.shape[0]

    a_cond = is_zero(matrix[0:nsize:2, 1:nsize:2])
    b_cond = is_zero(matrix[1:nsize:2, 0:nsize:2])
    c_cond = is_approx(matrix[0, :-1], matrix[1, 1:])
    d_cond = is_approx(matrix[-2, :-1], matrix[-1, 1:])

    return a_cond and b_cond and c_cond and d_cond


#######################################################
def get_global_phase(unitary):
    """ Get the global phase of arbitrary unitary, and get the special unitary.

    Args:
        unitary (np.array): arbitrary unitary
    Returns:
        global_phase: the global phase of arbitrary unitary
        special_unitary (np.array)
    """
    coefficient = np.linalg.det(unitary) ** (-0.5)
    global_phase = -cmath.phase(coefficient)
    special_unitary = coefficient * unitary
    return global_phase, special_unitary


def matrix_distance_squared(unitary1, unitary2):
    """ Used to compare the distance of two matrices. The global phase is ignored.

    Args:
        unitary1 (np.array): A unitary matrix in the form of a numpy ndarray.
        unitary2 (np.array): Another unitary matrix.

    Returns:
        Float : A single value between 0 and 1 indicating how closely unitary1 and unitary2 match.
        A value close to 0 indicates that unitary1 and unitary2 are the same unitary.
    """
    return np.abs(1 - np.abs(np.sum(np.multiply(unitary1, np.conj(unitary2)))) / unitary1.shape[0])
