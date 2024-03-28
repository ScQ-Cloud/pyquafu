#  (C) Copyright 2023 Beijing Academy of Quantum Information Sciences
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import copy
from abc import ABCMeta
from typing import Dict, Iterable, List

from quafu.elements import Instruction, QuantumGate, ControlledGate


class OracleGateMeta(ABCMeta):
    """
    Metaclass to create OracleGate CLASS which is its instance.
    """

    def __init__(cls, name, bases, attrs):
        for attr_name in ["cls_name", "gate_structure", "qubit_num"]:
            assert (
                attr_name in attrs
            ), f"OracleGateMeta: {attr_name} not found in {attrs}."

        # TODO: check if instructions inside gate_structure are valid

        super().__init__(name, bases, attrs)
        cls.name = attrs.__getitem__("cls_name")
        cls.gate_structure = attrs.__getitem__("gate_structure")
        cls.qubit_num = attrs.__getitem__("qubit_num")


class OracleGate(QuantumGate):  # TODO: Can it be related to OracleGateMeta explicitly?
    """
    OracleGate is a gate that can be customized by users.

    Attributes:
        name: name of the gate
        gate_structure: structure of the gate
        qubit_num: number of qubits the gate acts on
        insides: instances of instructions inside the gate from gate_structure
    """

    name = None
    gate_structure = []
    qubit_num = 0

    _named_pos = {}
    insides = []

    def __init__(self, pos: List, paras=None, label: str = None):
        """
        Args:
            pos: position of the gate
            paras: parameters of the gate  # TODO: how to set paras?
            label: label when draw or plot
        """
        if not self.qubit_num == len(pos):
            raise ValueError(
                f"OracleGate: qubit number {self.qubit_num} does not match pos length {len(pos)}."
            )
        super().__init__(pos=pos, paras=paras)

        self.__instantiate_gates__()
        self.label = label if label is not None else self.name

    @property
    def matrix(self):
        # TODO: this should be finished according to usage in simulation
        #       to avoid store very large matrix
        raise NotImplemented

    @property
    def named_pos(self) -> Dict:
        return {"pos": self.pos}

    @property
    def named_paras(self) -> Dict:
        # TODO: how to manage paras and the names?
        return self._named_pos

    def to_qasm(self, with_para):
        # TODO: this is similar to QuantumCircuit.to_qasm
        raise NotImplemented

    def __instantiate_gates__(self) -> None:
        """
        Instantiate the gate structure through coping ins and bit mapping.
        """
        bit_mapping = {i: p for i, p in enumerate(self.pos)}

        def map_pos(pos):
            if isinstance(pos, int):
                return bit_mapping[pos]
            elif isinstance(pos, Iterable):
                return [bit_mapping[p] for p in pos]
            else:
                raise ValueError

        for gate in self.gate_structure:
            gate_ = copy.deepcopy(gate)
            for key, val in gate.named_pos.items():
                setattr(gate_, key, map_pos(val))
            setattr(gate_, "pos", map_pos(gate.pos))
            self.insides.append(gate_)


class ControlledOracle(OracleGate):
    def __init__(self, ctrls: List, targs: List, paras=None, label: str = None):
        super().__init__(pos=ctrls + targs, paras=paras, label=label)
        self.ctrls = ctrls
        self.targs = targs
        self.__check_insides__()

    def __check_insides__(self):
        """Make sure that every gate inside is a controlled gate, and the control qubits correct."""
        for gate in self.insides:
            if not isinstance(gate, ControlledGate):
                raise ValueError(f"ControlledOracle: {gate} is not a controlled gate.")
            if not set(self.ctrls) in gate.ctrls:
                raise ValueError(
                    f"ControlledOracle: {gate} control qubits {gate.ctrls} does not match {self.ctrls}."
                )


def customize_gate(
    cls_name: str,
    gate_structure: List[Instruction],
    qubit_num: int,
    controlled: bool = False,
):
    """
    Helper function to create customized gate class

    Args:
        cls_name: name of the gate class
        gate_structure: a list of instruction INSTANCES
        qubit_num: number of qubits of the gate (TODO: extract from gate_structure?)
        controlled: whether the gate is a ctrl-U

    Returns:
        customized gate class

    Raises:
        ValueError: if gate class already exists
    """
    if cls_name in QuantumGate.gate_classes:
        raise ValueError(f"Gate class {cls_name} already exists.")

    attrs = {
        "cls_name": cls_name,
        "gate_structure": gate_structure,  # TODO: translate
        "qubit_num": qubit_num,
    }
    if controlled:
        customized_cls = type(cls_name, (ControlledOracle,), attrs)
    else:
        customized_cls = OracleGateMeta(cls_name, (OracleGate,), attrs)
    assert issubclass(customized_cls, OracleGate)
    QuantumGate.register_gate(customized_cls, cls_name)
    return customized_cls
