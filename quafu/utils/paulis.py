from functools import reduce

import numpy as np
import sparse

si = sparse.COO(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex))

sx = sparse.COO(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex))

sy = sparse.COO(np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex))

sz = sparse.COO(np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex))

spin = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]


def rx(phi):
    return np.array(
        [
            [np.cos(phi / 2), -1j * np.sin(phi / 2)],
            [-1j * np.sin(phi / 2), np.cos(phi / 2)],
        ]
    )


def ry(phi):
    return np.array(
        [[np.cos(phi / 2), -np.sin(phi / 2)], [np.sin(phi / 2), np.cos(phi / 2)]]
    )


def tensorl(ml):
    return reduce(sparse.kron, ml, 1)


def Nbit_single(N):
    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sx
        sx_list.append(tensorl(op_list))

        op_list[n] = sy
        sy_list.append(tensorl(op_list))

        op_list[n] = sz
        sz_list.append(tensorl(op_list))

    return sx_list, sy_list, sz_list
