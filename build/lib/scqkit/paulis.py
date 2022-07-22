import numpy as np
from functools import reduce

si = np.array([[1., 0.],
               [0., 1.]])

sx = np.array([[0., 1.],
               [1., 0.]])

sy = np.array([[0., -1.j],
               [1.j, 0.]]) 

sz = np.array([[1., 0.],    
               [0., -1.]])

spin = [np.array([1., 0.]), np.array([0., 1.])]

def rx(phi):
    return np.array([[np.cos(phi / 2), -1j * np.sin(phi / 2)],
                     [-1j * np.sin(phi / 2), np.cos(phi / 2)]])
            
def ry(phi):
    return np.array([[np.cos(phi / 2), -np.sin(phi / 2)],
                     [np.sin(phi / 2), np.cos(phi / 2)]])

def tensorl(ml):
    return reduce(np.kron, ml, 1)


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