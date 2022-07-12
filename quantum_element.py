#This is the file for abstract quantum gates class
from collections import Iterable


class Barrier(object):
    def __init__(self, pos):
        self.name = "barrier"
        self.__pos = pos
        
    @property
    def pos(self):
        return self.__pos
    
    @pos.setter
    def pos(self, pos):
        self.__pos = pos

    def to_QLisp(self):
        return ("Barrier", tuple(["Q%d" %i for i in self.pos]))


class QuantumGate(object):
    def __init__(self, name, pos, paras):
        self.__name = name
        self.__pos = pos
        self.__paras = paras

    @property
    def name(self):
        return self.__name
    
    @property
    def pos(self):
        return self.__pos
    
    @property
    def paras(self):
        return self.__paras

    @pos.setter
    def pos(self, pos):
        self.__pos = pos
    
    @paras.setter
    def paras(self, paras):
        self.__paras = paras

class SingleQubitGate(QuantumGate):
    def __init__(self, name, pos):
        super().__init__(name, pos, paras=None)

    def to_QLisp(self):
        return ((self.name, "Q%d" %self.pos))

    def to_nodes(self):
        return (1 ,self.name, 0, self.pos)
    
    def to_IOP(self):
        return [self.name, self.pos, 0.]

class ParaSingleQubitGate(QuantumGate):
    def __init__(self, name, pos, paras):
        super().__init__(name, pos, paras=paras)

    def to_QLisp(self):
        if isinstance(self.paras, Iterable):
            return ((self.name, *self.paras), "Q%d" %self.pos) 
        else: 
            return ((self.name, self.paras), "Q%d" %self.pos)

    def to_nodes(self):
        return (1, self.name, self.paras, self.pos)

    def to_IOP(self):
        if isinstance(self.paras, Iterable):
            return [self.name, self.pos, *self.paras]
        else:
            return [self.name, self.pos, self.paras]

class TwoQubitGate(QuantumGate):
    def __init__(self, name, pos):
        super().__init__(name, pos, paras=None)
        if not len(pos) == 2:
            raise ValueError("Two postion of a two-qubit gate should be provided")
    
    def to_QLisp(self):
        return (self.name, ("Q%d" %self.pos[0], "Q%d" %self.pos[1]))

    def to_nodes(self):
        return (2, self.name, self.pos[0], self.pos[1])
    
    def to_IOP(self):
        return [self.name, self.pos]

class ParaTwoQubitGate(QuantumGate):
    def __init__(self, name, pos, paras):
        super().__init__(name, pos, paras)
        if not len(pos) == 2:
            raise ValueError("Two postion of a two-qubit gate should be provided")

    def to_QLisp(self):
        if isinstance(self.paras, Iterable):
            return ((self.name, *self.paras), ("Q%d" %self.pos[0], "Q%d" %self.pos[1])) 
        else: 
            return ((self.name, self.paras), ("Q%d" %self.pos[0], "Q%d" %self.pos[1]))

    def to_nodes(self):
        return (2, self.name, self.paras, self.pos[0], self.pos[1])

    def to_IOP(self):
        if isinstance(self.paras, Iterable):
            return [self.name, self.pos, *self.paras]
        else:
            return [self.name, self.pos, self.paras]

class ControlGate(TwoQubitGate):
    def __init__(self, name, ctrl, targ):
        super().__init__(name, [ctrl, targ])
        self.__ctrl = ctrl
        self.__targ = targ
    
    @property
    def ctrl(self):
        return self.__ctrl
    
    @property
    def targ(self):
        return self.__targ

    @ctrl.setter
    def ctrl(self, ctrl):
        self.__ctrl = ctrl
    
    @targ.setter
    def targ(self, targ):
        self.__targ = targ

    def to_QLisp(self):
        return (self.name, ("Q%d" %self.ctrl, "Q%d" %self.targ))
    
    def to_nodes(self):
        return (2, self.name, self.ctrl, self.targ)
    
    def to_IOP(self):
        return [self.name, [self.ctrl, self.targ]]