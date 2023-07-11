import numpy as np

IdMatrix = np.eye(2, dtype=complex)
XMatrix = np.array([[0., 1.], [1., 0.]], dtype=complex)
YMatrix = np.array([[0., -1.j], [1.j, 0.]], dtype=complex)
ZMatrix = np.array([[1., 0.], [0., -1.]], dtype=complex)
SMatrix = np.array([[1., 0.], [0., 1.j]], dtype=complex)
SXMatrix = np.array([[1., 1.j], [1.j, 1.]], dtype=complex) / np.sqrt(2)
SYMatrix = np.array([[1., -1.], [1., 1.]], dtype=complex) / np.sqrt(2)
TMatrix = np.array([[1., 0.], [0., np.exp(1.j * np.pi / 4)]], dtype=complex)
WMatrix = (XMatrix + YMatrix) / np.sqrt(2)
SWMatrix = np.array([[0.5 + 0.5j, -np.sqrt(0.5) * 1j],
                     [np.sqrt(0.5), 0.5 + 0.5j]], dtype=complex)
HMatrix = (XMatrix + ZMatrix) / np.sqrt(2)
SwapMatrix = np.array([[1., 0., 0., 0.],
                       [0., 0., 1., 0.],
                       [0., 1., 0., 0.],
                       [0., 0., 0., 1.]], dtype=complex)
ISwapMatrix = np.array([[1., 0., 0., 0.],
                        [0., 0., 1.j, 0.],
                        [0., 1.j, 0., 0.],
                        [0., 0., 0., 1.]], dtype=complex)
CXMatrix = np.array([[1., 0., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 0., 0., 1.],
                     [0., 0., 1., 0.]], dtype=complex)
CYMatrix = np.array([[1., 0., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 0., 0., -1.j],
                     [0., 0., 1.j, 0.]], dtype=complex)
CZMatrix = np.array([[1., 0., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., -1.]], dtype=complex)
ToffoliMatrix = np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 1., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 1., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 1., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 1., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 1., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 1.],
                          [0., 0., 0., 0., 0., 0., 1., 0.]], dtype=complex)
FredkinMatrix = np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 1., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 1., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 1., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 1., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 1., 0.],
                          [0., 0., 0., 0., 0., 1., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 1.]], dtype=complex)


def u2matrix(_phi=0., _lambda=0.):
    """OpenQASM 3.0 specification"""
    return np.array([[1., np.exp(-1.j * _lambda)],
                     [np.exp(1.j * _phi), np.exp((_phi + _lambda) * 1.j)]], dtype=complex)


def u3matrix(_theta=0., _phi=0., _lambda=0.):
    """OpenQASM 3.0 specification"""
    return np.array([[np.cos(0.5 * _theta), -np.exp(_lambda * 1.j) * np.sin(0.5 * _theta)],
                     [np.exp(_phi * 1.j) * np.sin(0.5 * _theta),
                      np.exp((_phi + _lambda) * 1.j) * np.cos(0.5 * _theta)]], dtype=complex)


def rx_mat(theta):
    return np.array([[np.cos(0.5 * theta), -1.j * np.sin(0.5 * theta)],
                     [-1.j * np.sin(0.5 * theta), np.cos(0.5 * theta)]], dtype=complex)


def ry_mat(theta):
    return np.array([[np.cos(0.5 * theta), - np.sin(0.5 * theta)],
                     [np.sin(0.5 * theta), np.cos(0.5 * theta)]], dtype=complex)


def rz_mat(theta):
    return np.array([[np.exp(-0.5j * theta), 0.],
                     [0., np.exp(0.5j * theta)]], dtype=complex)


def pmatrix(labda):
    return np.array([[1, 0],
                     [0, np.exp(1j * labda)]], dtype=complex)


def rxx_mat(theta):
    """Unitary evolution of XX interaction"""
    return np.array([[np.cos(theta / 2), 0, 0, -1j * np.sin(theta / 2)],
                     [0, np.cos(theta / 2), -1j * np.sin(theta / 2), 0],
                     [0, -1j * np.sin(theta / 2), np.cos(theta / 2), 0],
                     [-1j * np.sin(theta / 2), 0, 0, np.cos(theta / 2)]
                     ])


def ryy_mat(theta):
    """ Unitary evolution of YY interaction"""
    c = np.cos(theta / 2)
    s = 1j * np.sin(theta / 2)
    return np.array([[c, 0, 0, +s],
                     [0, c, -s, 0],
                     [0, -s, c, 0],
                     [+s, 0, 0, c]
                     ])


def rzz_mat(theta):
    return np.array([[np.exp(-1j * theta / 2), 0, 0, 0],
                     [0, np.exp(1j * theta / 2), 0, 0],
                     [0, 0, np.exp(1j * theta / 2), 0],
                     [0, 0, 0, np.exp(-1j * theta / 2)]
                     ])

# def su2_matrix(gamma: float, beta: float, delta: float):
#     """
#     SU = Rz(beta)Ry(gamma)Rz(delta).
#
#     Symbol convention is the same as in the textbook
#     of Chuang and Nielsen.
#     """
#     s, c = np.sin(gamma / 2), np.cos(gamma / 2)
#     alpha1, alpha2 = (delta + beta) / 2, (delta - beta) / 2
#     su2_mat = np.array([[np.exp(-1.j * alpha1) * c, -np.exp(-1.j * alpha2) * s],
#                         [np.exp(1.j * alpha2) * s, np.exp(1.j * alpha1) * c]])
#     return su2_mat
#
#
# def u2_matrix(alpha: float, gamma: float, beta: float, delta: float):
#     return np.exp(1.j * alpha) * su2_matrix(gamma, beta, delta)
