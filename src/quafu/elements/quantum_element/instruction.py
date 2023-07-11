from abc import ABC, abstractmethod
from typing import Union, List


class Instruction(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError('name is not implemented for %s' % self.__class__.__name__
                                  + ', this should never happen.')
    # """
    # Metaclass for primitive instructions in DIRECTED ACYCLIC quantum circuits
    # or quantum-classical circuits.
    #
    # Member variables:
    #
    # - sd_name: standard name without args or extra annotations. Used for communications
    #   and identifications within pyquafu program, for example, to translate a qu_gate into qasm.
    #   Ideally every known primitive during computation should have such a name, and should be
    #   chosen as the most commonly accepted convention. NOT allowed to be changed by users once
    #   the class is instantiated.
    #
    # - pos: positions of relevant quantum or classical bits, legs of DAG.
    #
    # - dorder: depth order, or topological order of DAG
    #
    # - symbol: label that can be freely customized by users. If sd_name is not None, name is the
    #   same as sd_name by default. Otherwise, a symbol has to be specified while sd_name remains
    #   as None to indicate that this is a use-defined class.
    #  """
    #
    # _ins_id = None  # type: str
    #
    # def __init__(self, pos: PosType, label: str = None, paras: ParaType = None):
    #     # if pos is not iterable, make it be
    #     self.pos = [pos] if isinstance(pos, int) else pos
    #     if label:
    #         self.label = label
    #     else:
    #         if self._ins_id is None:
    #             raise ValueError('For user-defined instruction, label has to be specified.')
    #         self.label = self._ins_id
    #         if paras:
    #             self.label += '(' + ', '.join(['%.3f' % _ for _ in paras.values()]) + ')'
    #     self.paras = paras
    #
    # @classmethod
    # def get_ins_id(cls):
    #     return cls._ins_id
    #
    # @abstractmethod
    # def openqasm2(self) -> str:
    #     pass

    # @_ins_id.setter
    # def sd_name(self, name: str):
    #     if self.sd_name is None:
    #         self.sd_name = name
    #     else:
    #         import warnings
    #         warnings.warn(message='Invalid assignment, names of standard '
    #                               'instructions are not alterable.')

    # def to_dag_node(self):
    #     name = self.get_ins_id()
    #     label = self.__repr__()
    #
    #     pos = self.pos
    #     paras = self.paras
    #     paras = {} if paras is None else paras
    #     duration = paras.get('duration', None)
    #     unit = paras.get('unit', None)
    #     channel = paras.get('channel', None)
    #     time_func = paras.get('time_func', None)
    #
    #     return InstructionNode(name, pos, paras, duration, unit, channel, time_func, label)


PosType = Union[int, List[int]]
ParaType = dict[str, Union[float, int]]
