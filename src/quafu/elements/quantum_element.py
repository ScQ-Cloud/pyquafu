# This is the file for abstract quantum gates class
from typing import Union, Callable, List, Tuple, Iterable, Any, Optional
import numpy as np
from functools import reduce
import copy

class Barrier(object):
    def __init__(self, pos):
        self.name = "barrier"
        self.__pos = pos
        self.symbol = "||"

    @property
    def pos(self):
        return self.__pos

    @pos.setter
    def pos(self, pos):
        self.__pos = pos

    def __repr__(self):
        return f"{self.__class__.__name__}"

class Delay(object):
    def __init__(self, pos : int, duration : int, unit="ns"):
        self.name = "delay"
        if isinstance(duration, int):
            self.duration = duration
        else:
            raise TypeError("duration must be int")
        self.unit=unit
        self.pos=pos
        self.symbol = "Delay(%d%s)" %(duration, unit)

    def __repr__(self):
        return f"{self.__class__.__name__}"

class XYResonance(object):
    def __init__(self, qs : int, qe : int, duration : int, unit="ns"):
        self.name = "XY"
        if isinstance(duration, int):
            self.duration = duration
        else:
            raise TypeError("duration must be int")
        self.unit=unit
        self.pos=list(range(qs, qe+1))
        self.symbol = "XY(%d%s)" %(duration, unit)

class QuantumGate(object):
    def __init__(self, name: str, pos: Union[int, List[int]], paras: Union[None,float, List[float]], matrix):
        self.name = name
        self.pos = pos
        self.paras = paras
        self.matrix = matrix
        
        if paras:
            if isinstance(paras, Iterable):
                self.symbol = "%s(" %self.name + ",".join(["%.3f" %para for para in self.paras]) + ")"
            else:
                self.symbol = "%s(%.3f)" % (self.name, paras)
        else:
            self.symbol = "%s" %self.name

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, _name):
        self.__name = _name

    @property
    def pos(self):
        return self.__pos

    @pos.setter
    def pos(self, _pos):
        self.__pos = _pos

    @property
    def paras(self):
        return self.__paras

    @paras.setter
    def paras(self, _paras):
        self.__paras = _paras

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        self._matrix = matrix

    def __str__(self):
        properties_names = ['pos', 'paras', 'matrix']
        properties_values = [getattr(self, x) for x in properties_names]
        return "%s:\n%s" % (self.__class__.__name__, '\n'.join(
            [f"{x} = {repr(properties_values[i])}" for i, x in enumerate(properties_names)]))

    def __repr__(self):
        return f"{self.__class__.__name__}"


class SingleQubitGate(QuantumGate):
    def __init__(self, name: str, pos: int, paras, matrix):
        super().__init__(name, pos, paras=paras, matrix=matrix)

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        if isinstance(matrix, (np.ndarray, List)):
            if np.shape(matrix) == (2, 2):
                self._matrix = np.asarray(matrix, dtype=complex)
            else:
                raise ValueError(f'`{self.__class__.__name__}.matrix.shape` must be (2, 2)')
        elif isinstance(matrix, type(None)):
            self._matrix = matrix
        else:
            raise TypeError("Unsupported `matrix` type")

    def get_targ_matrix(self, reverse_order=False):
        return self.matrix

class FixedSingleQubitGate(SingleQubitGate):
    def __init__(self, name, pos, matrix):
        super().__init__(name, pos, paras=None, matrix=matrix)


class ParaSingleQubitGate(SingleQubitGate):
    def __init__(self, name, pos, paras, matrix):
        super().__init__(name, pos, paras=paras, matrix=matrix)

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        if isinstance(matrix, Callable):
            self._matrix = matrix(self.paras)
        elif isinstance(matrix, (np.ndarray, List)):
            if np.shape(matrix) == (2, 2):
                self._matrix = np.asarray(matrix, dtype=complex)
            else:
                raise ValueError(f'`{self.__class__.__name__}.matrix.shape` must be (2, 2)')
        elif isinstance(matrix, type(None)):
            self._matrix = matrix
        else:
            raise TypeError("Unsupported `matrix` type")


class MultiQubitGate(QuantumGate):
    def __init__(self, name: str, pos: List[int], paras, matrix):
        super().__init__(name, pos, paras=paras, matrix=matrix)

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        if isinstance(matrix, np.ndarray):
            self._matrix = matrix
        elif isinstance(matrix, List):
            self._matrix = np.array(matrix, dtype=complex)
        else:
            raise TypeError("Unsupported `matrix` type")

        self.reorder_matrix()

    def reorder_matrix(self):
        """Reorder the input sorted matrix to the pos order """
        qnum = len(self.pos)
        dim = 2**qnum
        inds = np.argsort(self.pos)
        inds = np.concatenate([inds, inds+qnum])
        tensorm = self._matrix.reshape([2]*2*qnum)
        self._matrix = np.transpose(tensorm, inds).reshape([dim, dim])

    def get_targ_matrix(self, reverse_order=False):
        targ_matrix = self._matrix
        if reverse_order and (len(self.pos) > 1):
            qnum = len(self.pos)
            order = np.array(range(len(self.pos))[::-1])
            order = np.concatenate([order, order+qnum])
            dim = 2**qnum
            tensorm = self._matrix.reshape([2]*2*qnum)
            targ_matrix = np.transpose(tensorm, order).reshape([dim, dim])

        return targ_matrix
         

class FixedMultiQubitGate(MultiQubitGate):
    def __init__(self, name: str, pos: List[int], matrix):
        super().__init__(name, pos, paras=None, matrix=matrix)


class ParaMultiQubitGate(MultiQubitGate):
    def __init__(self, name, pos, paras, matrix):
        super().__init__(name, pos, paras, matrix=matrix)

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        if isinstance(matrix, Callable):
            self._matrix = matrix(self.paras)
            self.reorder_matrix()
        elif isinstance(matrix, (np.ndarray, List)):
            self._matrix = matrix
            self.reorder_matrix()
        else:
            raise TypeError("Unsupported `matrix` type")
        

class ControlledGate(MultiQubitGate):
    """ Controlled gate class, where the matrix act non-trivallly on target qubits"""
    def __init__(self, name, targe_name, ctrls: List[int], targs: List[int], paras, matrix):
        self.ctrls = ctrls
        self.targs = targs
        self.targ_name = targe_name
        super().__init__(name, ctrls+targs, paras, matrix)
        self._targ_matrix = matrix

        if paras:
            if isinstance(paras, Iterable):
                self.symbol = "%s(" %self.targ_name + ",".join(["%.3f" %para for para in self.paras]) + ")"
            else:
                self.symbol = "%s(%.3f)" % (self.targ_name, paras)
        else:
            self.symbol = "%s" %self.targ_name

            
    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix : Union[np.ndarray, Callable]):
        targ_dim = 2**(len(self.targs))
        qnum = len(self.pos)
        dim = 2**(qnum)
        if isinstance(matrix, Callable):
            matrix = matrix(self.paras)

        if matrix.shape[0] != targ_dim:
            raise ValueError("Dimension dismatch")
        else:
            self._matrix = np.zeros((dim , dim), dtype=complex)
            control_dim = 2**len(self.pos) - targ_dim
            for i in range(control_dim):
                self._matrix[i, i] = 1.
            
            self._matrix[control_dim:, control_dim:] = matrix
            self.reorder_matrix()
            self._targ_matrix = matrix

    def get_targ_matrix(self, reverse_order=False):
        return self._targ_matrix

class ControlledU(ControlledGate):
    def __init__(self, name, ctrls: List[int], U: Union[SingleQubitGate, MultiQubitGate]):
        self.targ_gate = U
        targs = U.pos
        if isinstance(targs, int):
            targs = [targs]
        
        super().__init__(name, U.name, ctrls, targs, U.paras, matrix=self.targ_gate.matrix)
    
    def get_targ_matrix(self, reverse_order=False):
        return self.targ_gate.get_targ_matrix(reverse_order)



