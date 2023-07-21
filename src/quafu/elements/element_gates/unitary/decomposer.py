import cmath
import math

import numpy as np
import scipy
from numpy import ndarray

from ..matrices import rz_mat, ry_mat, CXMatrix
from ..matrices import mat_utils as mu


class UnitaryDecomposer(object):
    def __init__(self, array: ndarray, qubits, verbose: bool = False):
        self.array = array
        self.qubit_num = len(qubits)
        self._check_unitary(array, self.qubit_num)

        self.qubits = qubits
        self.verbose = verbose

        self.Mk_table = genMk_table(self.qubit_num)  # initialize the general M^k lookup table
        self.gate_list = []

    def __call__(self, *args, **kwargs):
        _matrix = self.array
        self._decompose_matrix(_matrix, self.qubits)
        if self.verbose:
            print("Decomposition done.")

    @staticmethod
    def _check_unitary(array: ndarray, qubits):
        assert len(array.shape) == 2
        assert mu.is_unitary(array)
        assert len(array) == 2 ** qubits

    def _decompose_matrix(self, _matrix, qubits):
        qubit_num = len(qubits)
        self._check_unitary(_matrix, qubit_num)
        if qubit_num == 1:
            # self.gates_list.append((_matrix, qubits, 'U'))
            # ZYZ decomposition for single-qubit gate
            gamma, beta, alpha, global_phase = zyz_decomposition(_matrix)
            self.gate_list.append((rz_mat(gamma), qubits, 'RZ', gamma))
            self.gate_list.append((ry_mat(beta), qubits, 'RY', beta))
            self.gate_list.append((rz_mat(alpha), qubits, 'RZ', alpha))
            return None

        # if num_qubit == 2:
        # self.circuit.append((_matrix, qubits))
        #    print("Two qubit gate do not need to decompose.")
        # zyz_decomp(matrix)
        U00, U01, U10, U11 = mu.split_matrix(_matrix)

        # if bottomLeftCorner(n)==0 and topRightCorner(n)==0:
        if mu.is_zero(U01) and mu.is_zero(U10):
            if mu.is_approx(U00, U11):
                if self.verbose:
                    print('Optimization: Unitaries are equal, '
                          'skip one step in the recursion for unitaries of size')
                self._decompose_matrix(U00, qubits[1:])
            else:
                if self.verbose:
                    print("Optimization: q2 is zero, only demultiplexing will be performed.")
                V, D, W = demultiplexing(U00, U11)
                self._decompose_matrix(W, qubits[1:])
                self.multi_controlled_z(D, qubits[1:], qubits[0])
                self._decompose_matrix(V, qubits[1:])
        # Check if the kronecker product of a bigger matrix and the identity matrix.
        # By checking if the first row is equal to the second row one over, and if the last two rows are equal
        # Which means the last qubit is not affected by this gate
        elif mu.is_kron_with_id2(_matrix):
            # print("The last qubits not affect.")
            nsize = len(_matrix)
            self._decompose_matrix(_matrix[0:nsize:2, 0:nsize:2], qubits[:-1])
        else:
            # print("CSD decomposition.")
            L0, L1, R0, R1, c, s = fat_csd(_matrix)
            V, D, W = demultiplexing(R0, R1)

            self._decompose_matrix(W, qubits[1:])
            self.multi_controlled_z(D, qubits[1:], qubits[0])
            self._decompose_matrix(V, qubits[1:])
            self.multi_controlled_y(s, qubits[1:], qubits[0])

            V, D, W = demultiplexing(L0, L1)
            self._decompose_matrix(W, qubits[1:])
            self.multi_controlled_z(D, qubits[1:], qubits[0])
            self._decompose_matrix(V, qubits[1:])

    def multi_controlled_z(self, D, qubits, target_qubit):
        print(D.shape[0])
        assert len(qubits) == int(math.log(D.shape[0], 2))
        num_qubit = len(qubits)
        # print("The size of D matrix {}".format(len(qubits)))
        # print(qubits)

        # alphas = -2*1j*np.log(np.diag(D))
        alphas = 2 * 1j * np.log(np.diag(D))
        Mk = self.Mk_table[num_qubit - 1]
        thetas = np.linalg.solve(Mk, alphas)
        # print(thetas)
        assert len(thetas) == 2 ** num_qubit

        index = get_multi_control_index(num_qubit)
        assert len(index) == len(thetas)

        for i in range(len(index)):
            control_qubit = qubits[index[i]]
            self.gate_list.append((rz_mat(thetas[i]), [target_qubit], 'RZ', thetas[i]))
            self.gate_list.append((CXMatrix, [control_qubit, target_qubit], 'CX'))

    def multi_controlled_y(self, ss, qubits, target_qubit):
        assert len(qubits) == int(math.log(ss.shape[0], 2))
        num_qubit = len(qubits)

        alphas = -2 * np.arcsin(np.diag(ss))
        Mk = self.Mk_table[num_qubit - 1]
        thetas = np.linalg.solve(Mk, alphas)
        # print(thetas)
        assert len(thetas) == 2 ** num_qubit

        index = get_multi_control_index(num_qubit)
        assert len(index) == len(thetas)

        for i in range(len(index)):
            control_qubit = qubits[index[i]]
            self.gate_list.append((ry_mat(thetas[i]), [target_qubit], 'RY', thetas[i]))
            self.gate_list.append((CXMatrix, [control_qubit, target_qubit], 'CX'))

    def apply_to_qc(self, qc):
        if len(self.gate_list) == 0:
            self()

        for g in self.gate_list:
            if g[2] == 'CX':
                qc.cnot(g[1][0], g[1][1])
            elif g[2] == 'RY':
                qc.ry(g[1][0], g[3].real)
            elif g[2] == 'RZ':
                qc.rz(g[1][0], g[3].real)
            elif g[2] == 'U':
                gamma, beta, alpha, global_phase = zyz_decomposition(g[0])
                qc.rz(g[1][0], gamma)
                qc.ry(g[1][0], beta)
                qc.rz(g[1][0], alpha)
            else:
                raise Exception("Unknown gate type or incorrect str: {}".format(g[2]))

        return qc


def zyz_decomposition(unitary):
    """ ZYZ decomposition of arbitrary single-qubit gate (unitary).
    SU = Rz(gamma) * Ry(beta) * Rz(alpha)

    Args:
        unitary (np.array): arbitrary unitary
    Returns:
        global_phase: the global phase of arbitrary unitary
        special_unitary (np.array)
    """
    if unitary.shape[0] == 2:
        global_phase, special_unitary = mu.get_global_phase(unitary)
        beta = 2 * math.atan2(abs(special_unitary[1, 0]), abs(special_unitary[0, 0]))
        t1 = cmath.phase(special_unitary[1, 1])
        t2 = cmath.phase(special_unitary[1, 0])
        alpha = t1 + t2
        gamma = t1 - t2
    else:
        raise Exception("ZYZ decomposition only applies to single-qubit gate.")
    return gamma, beta, alpha, global_phase


# # # # # # # Cosine-Sine Decomposition # # # # # # # # # # # #
def _thin_csd(q1, q2):
    p = q1.shape[0]
    print("the size of q1/q2: {}".format(p))

    u1, c, v1 = np.linalg.svd(q1)
    v1d = np.conjugate(v1).T
    c = np.flip(c)

    cm = np.zeros((p, p), dtype=complex)
    np.fill_diagonal(cm, c)

    u1 = np.fliplr(u1)
    v1d = np.fliplr(v1d)

    q2 = q2 @ v1d

    # find the biggest index of c[k] <= 1/np.sqrt(2)
    k = 0
    for i in range(1, p):
        if c[i] <= 1 / np.sqrt(2):
            k = i

    k = k + 1
    print("the k size: {}".format(k))

    u2, _ = np.linalg.qr(q2[:, 0:k], mode='complete')
    # u2, _= np.linalg.qr(q2, mode='complete')
    print("the size of u2: {}".format(u2.shape))
    # print("the u2 matrix: {}".format(u2))
    s = np.conjugate(u2).T @ q2
    print("the size of s: {}".format(s.shape))
    # print("the s matrix: {}".format(np.real(s)))

    if k < p:
        r2 = s[k:p, k:p]
        print("the size of rs: {}".format(r2.shape))
        ut, ss, vt = np.linalg.svd(r2)
        vtd = np.conjugate(vt).T
        s[k:p, k:p] = np.diag(ss)
        cm[:, k:p] = cm[:, k:p] @ vtd
        u2[:, k:p] = u2[:, k:p] @ ut
        v1d[:, k:p] = v1d[:, k:p] @ vtd

        w = cm[k:p, k:p]
        z, r = np.linalg.qr(w, mode='complete')
        cm[k:p, k:p] = r
        u1[:, k:p] = u1[:, k:p] @ z

    for i in range(p):
        if np.real(cm[i, i]) < 0:
            cm[i, i] = -cm[i, i]
            u1[:, i] = -u1[:, i]
        if np.real(s[i, i]) < 0:
            s[i, i] = -s[i, i]
            u2[:, i] = -u2[:, i]

    return u1, u2, v1d, cm, s


def fat_csd(matrix):
    """
    U = [U00, U01] = [u1    ][c  s][v1  ]
        [U10, U11] = [    u2][-s c][   v2]
    """
    # print(matrix)
    U00, U01, U10, U11 = mu.split_matrix(matrix)

    L0, L1, R0, cc, ss = _thin_csd(U00, U10)
    R0 = np.conjugate(R0).T
    ss = -ss

    # get the v2
    R1 = np.zeros_like(R0)
    p = R1.shape[0]
    for j in range(p):
        if np.abs(ss[j, j]) > np.abs(cc[j, j]):
            L0d = np.conjugate(L0).T
            tmp = L0d @ U01
            R1[j, :] = tmp[j, :] / ss[j, j]
        else:
            L1d = np.conjugate(L1).T
            tmp = L1d @ U11
            R1[j, :] = tmp[j, :] / cc[j, j]

    assert mu.is_approx(L0 @ cc @ R0, U00)
    assert mu.is_approx(-L1 @ ss @ R0, U10)
    assert mu.is_approx(L0 @ ss @ R1, U01)
    assert mu.is_approx(L1 @ cc @ R1, U11)

    zeros_m = np.zeros_like(L0)
    L = mu.stack_matrices(L0, zeros_m, zeros_m, L1)
    D = mu.stack_matrices(cc, ss, -ss, cc)
    R = mu.stack_matrices(R0, zeros_m, zeros_m, R1)
    assert mu.is_approx(matrix, L @ D @ R)

    return L0, L1, R0, R1, cc, ss  # L0, L1 is unitary


# # # # # # # # # # # # # # # # # # #  # # # # # # #
def demultiplexing(u1, u2):
    """
    U = [U1  0] = [V   0][D  0][W  0]
        [0  U2] = [0   V][0 D*][0  W]
    """
    assert mu.is_unitary(u1)
    assert mu.is_unitary(u2)

    u2_dg = np.conjugate(u2).T

    [d2, v] = scipy.linalg.schur(u1 @ u2_dg)
    assert mu.is_diagonal(d2)
    assert mu.is_approx(v @ d2 @ np.conjugate(v).T, u1 @ u2_dg)

    d_tmp = np.sqrt(np.diag(d2))
    d = np.diag(d_tmp)

    assert mu.is_approx(d @ d, d2)
    v_dg = np.conjugate(v).T

    w = d @ v_dg @ u2

    assert mu.is_approx(u1, v @ d @ w)
    assert mu.is_approx(u2, v @ np.conjugate(d).T @ w)

    zm = np.zeros_like(w)
    vv = mu.stack_matrices(v, zm, zm, v)
    dd = mu.stack_matrices(d, zm, zm, np.conjugate(d).T)
    ww = mu.stack_matrices(w, zm, zm, w)
    uu = mu.stack_matrices(u1, zm, zm, u2)

    assert mu.is_approx(vv @ dd @ ww, uu)
    assert mu.is_unitary(v)
    return v, d, w


def _graycode(n):
    for i in range(1 << n):
        gray = i ^ (i >> 1)
        yield "{0:0{1}b}".format(gray, n)


def get_multi_control_index(n):
    gray_codes = list(_graycode(n))
    size = 2 ** n

    index_list = []
    for i in range(size):
        str1 = gray_codes[i]
        str2 = gray_codes[(i + 1) % size]

        tmp = [k for k in range(len(str1)) if str1[k] != str2[k]]
        assert len(tmp) == 1
        index_list.append(tmp[0])

    return index_list


def genMk_table(nqubits):
    """
    TODO: add docstring
    """
    import re

    def bin2gray(num):
        return num ^ int((num >> 1))

    def genMk(k):

        Mk = np.zeros((2 ** k, 2 ** k))
        for i in range(2 ** k):
            for j in range(2 ** k):
                p = i & bin2gray(j)
                strbin = "{0:b}".format(p)
                tmp = [m.start() for m in re.finditer('1', strbin)]
                Mk[i, j] = (-1) ** len(tmp)

        return Mk

    genMk_lookuptable = []
    for n in range(1, nqubits + 1):
        tmp = genMk(n)
        genMk_lookuptable.append(tmp)

    return genMk_lookuptable
