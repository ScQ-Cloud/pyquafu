# (C) Copyright 2023 Beijing Academy of Quantum Information Sciences
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from contextlib import contextmanager
from typing import Any, Iterable, List, Optional

import numpy as np
import quafu.elements.element_gates as qeg
from quafu.elements import Measure, Reset, UnitaryDecomposer
from quafu.elements.classical_element import Cif
from quafu.elements.instruction import Instruction
from quafu.elements.parameters import Parameter, ParameterExpression, ParameterType

from ..elements import (
    Barrier,
    CircuitWrapper,
    ControlledCircuitWrapper,
    ControlledGate,
    ControlledOracle,
    Delay,
    KrausChannel,
    OracleGate,
    QuantumGate,
    QuantumPulse,
    XYResonance,
)
from ..elements.quantum_gate import CircuitWrapper, ControlledCircuitWrapper
from ..exceptions import CircuitError
from .classical_register import ClassicalRegister
from .quantum_register import QuantumRegister


class QuantumCircuit:
    """
    Representation of quantum circuit.
    """

    def __init__(self, qnum: int, cnum: Optional[int] = None, name="", *args, **kwargs):
        """
        Initialize a QuantumCircuit object

        Args:
            qnum (int): Total qubit number used
            cnum (int): Classical bit number, equals to qubit number in default
        """
        self.name = name
        self.qregs = [QuantumRegister(qnum)] if qnum > 0 else []
        cnum = self.num if cnum is None else cnum
        self.cregs = [ClassicalRegister(cnum)] if cnum > 0 else []
        self._gates = []
        self.instructions = []
        self.openqasm = ""
        self.circuit = []
        self._measures = []
        self.executable_on_backend = True
        self._used_qubits = []
        self._parameterized_gates = []
        self._parameter_grads = {}
        self._variables = []
        self._has_wrap = False

    @property
    def parameterized_gates(self):
        """Return the list of gates which the parameters are tunable"""
        if not self._parameterized_gates:
            self._parameterized_gates = [g for g in self.gates if len(g.paras) != 0]
        return self._parameterized_gates

    @property
    def num(self):
        return sum([len(qreg) for qreg in self.qregs])

    @num.setter
    def num(self, num: int):
        self.qregs = [QuantumRegister(num)] if num > 0 else []
        self.cregs = [ClassicalRegister(num)] if num > 0 else []

    @property
    def cbits_num(self):
        return sum([len(creg) for creg in self.cregs])

    @property
    def used_qubits(self) -> List:
        self.layered_circuit()
        return self._used_qubits

    @property
    def measures(self):
        measures = {}
        for meas in self._measures:
            measures.update(dict(zip(meas.qbits, meas.cbits)))
        return measures

    @measures.setter
    def measures(self, measures: dict):
        self._measures = [Measure(measures)]

    @property
    def gates(self):
        """Deprecated warning: due to historical reason, ``gates`` contains not only instances of
        QuantumGate, meanwhile not contains measurements. This attributes might be deprecated in
        the future. Better to use ``instructions`` which contains all the instructions.
        """
        return self._gates

    @gates.setter
    def gates(self, gates: list):
        self._gates = gates

    def __lshift__(self, operation: Instruction):
        max_pos = (
            max(operation.pos) if isinstance(operation.pos, Iterable) else operation.pos
        )
        if max_pos >= self.num:
            raise CircuitError("Operation act on qubit that not allocated")
        self.add_ins(operation)
        if isinstance(operation, CircuitWrapper):
            self._has_wrap = True
        return self

    # TODO(qtzhuang): add_gates is just a temporary call function to add gate from gate_list
    def add_gates(self, gates: list):
        for gate in gates:
            self.add_ins(gate)

    def add_gate(self, gate: QuantumGate):
        """
        Add quantum gate to circuit, with some checking.
        """
        pos = np.array(gate.pos)
        if np.any(pos >= self.num):
            raise CircuitError(f"Gate position out of range: {gate.pos}")
        self.gates.append(gate)

    def add_pulse(self, pulse: QuantumPulse, pos: int = None) -> "QuantumCircuit":
        """
        Add quantum gate from pulse.
        """
        if pos is not None:
            pulse.set_pos(pos)
        self.add_ins(pulse)
        return self

    def add_ins(self, ins: Instruction):
        """
        Add instruction to circuit, with NO checking yet.
        """
        if isinstance(ins, (QuantumGate, Delay, Barrier, XYResonance, KrausChannel)):
            # TODO: Delay, Barrier added by add_gate for backward compatibility.
            #       Figure out better handling in the future.
            self.add_gate(ins)
        self.instructions.append(ins)

    def add_noise(self, channel: str, channel_args, qubits=[], gates=[]):
        if channel not in ["bitflip", "dephasing", "depolarizing", "ampdamping"]:
            raise ValueError("Invalid channel name")

        for g in gates:
            if g not in list(QuantumGate.gate_classes.keys()):
                raise ValueError("Invalid gate name")

        newinstructions = []
        newgates = []
        for op in self.instructions:
            newinstructions.append(op)
            if isinstance(op, (QuantumGate, Delay, Barrier, XYResonance)):
                newgates.append(op)
            if isinstance(op, QuantumGate):
                add_q = False
                add_g = False
                if qubits:
                    for q in op.pos:
                        if q in qubits:
                            add_q = True
                else:
                    add_q = True

                if gates:
                    if op.name.lower() in gates:
                        add_g = True
                else:
                    add_g = True

                if add_q and add_g:
                    for q in op.pos:
                        if q in qubits:
                            newinstructions.append(
                                Instruction.ins_classes[channel](q, *channel_args)
                            )
                            newgates.append(
                                Instruction.ins_classes[channel](q, *channel_args)
                            )
        self.instructions = newinstructions
        self._gates = newgates
        return self

    @property
    def noised(self):
        if self._has_wrap:
            self.unwrap()

        for op in self.instructions:
            if isinstance(op, KrausChannel):
                return True
        return False

    def get_parameter_grads(self):
        if self._has_wrap:
            print(
                "warning: The circuit has wrapped gates, it will unwarp automaticllay"
            )
            self.unwrap()
        self._parameter_grads = {}
        for i, op in enumerate(self.gates):
            for j, para in enumerate(op.paras):
                if isinstance(para, Parameter):
                    if para not in self._parameter_grads.keys():
                        self._parameter_grads[para] = [[(i, j), 1.0]]
                    else:
                        self._parameter_grads[para].append([(i, j), 1.0])

                elif isinstance(para, ParameterExpression):
                    para_grads = para.grad()
                    for var in para._variables:
                        if var not in self._parameter_grads.keys():
                            self._parameter_grads[var] = [
                                [(i, j), para_grads[para._variables[var]]]
                            ]
                        else:
                            self._parameter_grads[var].append(
                                [(i, j), para_grads[para._variables[var]]]
                            )
        self._variables = list(self._parameter_grads.keys())
        return self._parameter_grads

    def _calc_parameter_grads(self):
        """
        Update parameter gradient value, not variables
        """
        for var, grads in self._parameter_grads.items():
            for item in grads:
                i, j = item[0]
                para = self.gates[i].paras[j]
                if isinstance(para, ParameterExpression):
                    para_grads = para.grad()
                    item[1] = para_grads[para._variables[var]]

        return self._parameter_grads

    @property
    def variables(self):
        self._variables = list(self.get_parameter_grads().keys())
        return self._variables

    def _update_params(self, values, order=[]):
        """
        Update variables' value, not variables
        Args:
            values: list of variables' value.
            order: For transplied circuit that change the order of variables,
            need pass the order to match untranspiled circuit's variable.
        """

        for i in range(len(values)):
            val = values[order[i]] if order else values[i]
            self._variables[i].value = val

    def update_params(self, paras_list: List[Any]):
        """Update parameters of parameterized gates
        Args:
            paras_list (List[Any]): list of params

        Raise:
            CircuitError
        """
        if len(paras_list) != len(self.parameterized_gates):
            raise CircuitError(
                "`params_list` must have the same size with parameterized gates"
            )

        # TODO(): Support updating part of params of a single gate
        for gate, paras in zip(self.parameterized_gates, paras_list):
            gate.update_params(paras)

    def layered_circuit(self) -> np.ndarray:
        """
        Make layered circuit from the gate sequence self.gates.

        Returns:
            A layered list with left justed circuit.
        """
        num = self.num
        gatelist = self.gates
        gateQlist = [[] for i in range(num)]
        used_qubits = []
        for gate in gatelist:
            if isinstance(gate, Delay) or isinstance(gate, QuantumPulse):
                gateQlist[gate.pos].append(gate)
                if gate.pos not in used_qubits:
                    used_qubits.append(gate.pos)

            else:
                pos1 = min(gate.pos)
                pos2 = max(gate.pos)
                gateQlist[pos1].append(gate)
                for j in range(pos1 + 1, pos2 + 1):
                    gateQlist[j].append(None)

                for pos in gate.pos:
                    if pos not in used_qubits:
                        used_qubits.append(pos)

                maxlayer = max([len(gateQlist[j]) for j in range(pos1, pos2 + 1)])
                for j in range(pos1, pos2 + 1):
                    layerj = len(gateQlist[j])
                    pos = layerj - 1
                    if not layerj == maxlayer:
                        for i in range(abs(layerj - maxlayer)):
                            gateQlist[j].insert(pos, None)

        # Add support of used_qubits for Reset and Cif
        def get_used_qubits(instructions):
            used_q = []
            for ins in instructions:
                if isinstance(ins, Cif):
                    used_q_h = get_used_qubits(ins.instructions)
                    for pos in used_q_h:
                        if pos not in used_q:
                            used_q.append(pos)
                elif isinstance(ins, Barrier):
                    continue
                elif isinstance(ins.pos, int):
                    if ins.pos not in used_q:
                        used_q.append(ins.pos)
                elif isinstance(ins.pos, list):
                    for pos in ins.pos:
                        if pos not in used_q:
                            used_q.append(pos)
            return used_q

        # Only consider of reset and cif
        for ins in self.instructions:
            if isinstance(ins, (Reset, Cif)):
                used_q = get_used_qubits([ins])
                for pos in used_q:
                    if pos not in used_qubits:
                        used_qubits.append(pos)

        maxdepth = max([len(gateQlist[i]) for i in range(num)])

        for gates in gateQlist:
            gates.extend([None] * (maxdepth - len(gates)))

        for m in self.measures.keys():
            if m not in used_qubits:
                used_qubits.append(m)
        used_qubits = np.sort(used_qubits)

        new_gateQlist = []
        for old_qi in range(len(gateQlist)):
            gates = gateQlist[old_qi]
            if old_qi in used_qubits:
                new_gateQlist.append(gates)

        lc = np.array(new_gateQlist)
        lc = np.vstack((used_qubits, lc.T)).T
        self.circuit = lc
        self._used_qubits = list(used_qubits)
        return self.circuit

    def draw_circuit(self, width: int = 4, return_str: bool = False):
        """
        Draw layered circuit using ASCII, print in terminal.
        Args:
            width (int): The width of each gate.
            return_str: Whether return the circuit string.
        """
        # TODO: support draw other instructions
        self.layered_circuit()
        gateQlist = self.circuit
        num = gateQlist.shape[0]
        depth = gateQlist.shape[1] - 1
        printlist = np.array([[""] * depth for i in range(2 * num)], dtype="<U30")

        reduce_map = dict(zip(gateQlist[:, 0], range(num)))
        reduce_map_inv = dict(zip(range(num), gateQlist[:, 0]))
        for l in range(depth):
            layergates = gateQlist[:, l + 1]
            maxlen = 1 + width
            for i in range(num):
                gate = layergates[i]
                if isinstance(gate, Barrier):
                    pos = [i for i in gate.pos if i in reduce_map.keys()]
                    q1 = reduce_map[min(pos)]
                    q2 = reduce_map[max(pos)]
                    printlist[2 * q1 : 2 * q2 + 1, l] = "||"
                    maxlen = max(maxlen, len("||"))
                elif isinstance(gate, Instruction) and len(gate.pos) == 1:
                    printlist[i * 2, l] = gate.symbol
                    maxlen = max(maxlen, len(gate.symbol) + width)

                elif isinstance(gate, Instruction) and len(gate.pos) > 1:
                    q1 = reduce_map[min(gate.pos)]
                    q2 = reduce_map[max(gate.pos)]
                    printlist[2 * q1 + 1 : 2 * q2, l] = "|"
                    printlist[q1 * 2, l] = "#"
                    printlist[q2 * 2, l] = "#"
                    if isinstance(gate, ControlledGate):  # Controlled-Multiqubit gate
                        for ctrl in gate.ctrls:
                            printlist[reduce_map[ctrl] * 2, l] = "*"

                        if gate._targ_name == "SWAP":
                            printlist[reduce_map[gate.targs[0]] * 2, l] = "x"
                            printlist[reduce_map[gate.targs[1]] * 2, l] = "x"
                        else:
                            tq1 = reduce_map[min(gate.targs)]
                            tq2 = reduce_map[max(gate.targs)]
                            printlist[tq1 * 2, l] = "#"
                            printlist[tq2 * 2, l] = "#"
                            if tq1 + tq2 in [
                                reduce_map[ctrl] * 2 for ctrl in gate.ctrls
                            ]:
                                printlist[tq1 + tq2, l] = "*" + gate.symbol
                            else:
                                printlist[tq1 + tq2, l] = gate.symbol
                            maxlen = max(maxlen, len(gate.symbol) + width)

                    else:  # Multiqubit gate
                        if gate.name == "SWAP":
                            printlist[q1 * 2, l] = "x"
                            printlist[q2 * 2, l] = "x"

                        else:
                            printlist[q1 + q2, l] = gate.symbol
                            maxlen = max(maxlen, len(gate.symbol) + width)

            printlist[-1, l] = maxlen

        circuitstr = []
        for j in range(2 * num - 1):
            if j % 2 == 0:
                linestr = ("q[%d]" % (reduce_map_inv[j // 2])).ljust(6) + "".join(
                    [
                        printlist[j, l].center(int(printlist[-1, l]), "-")
                        for l in range(depth)
                    ]
                )
                if reduce_map_inv[j // 2] in self.measures.keys():
                    linestr += " M->c[%d]" % self.measures[reduce_map_inv[j // 2]]
                circuitstr.append(linestr)
            else:
                circuitstr.append(
                    "".ljust(6)
                    + "".join(
                        [
                            printlist[j, l].center(int(printlist[-1, l]), " ")
                            for l in range(depth)
                        ]
                    )
                )
        circuitstr = "\n".join(circuitstr)

        if return_str:
            return circuitstr
        else:
            print(circuitstr)

    def plot_circuit(self, *args, **kwargs):
        from quafu.visualisation.circuitPlot import CircuitPlotManager

        cmp = CircuitPlotManager(self)
        return cmp(*args, **kwargs)

    def from_openqasm(self, openqasm: str):
        """
        Initialize the circuit from openqasm text.
        Args:
            openqasm: input openqasm str.
        """
        from quafu.qfasm.qfasm_convertor import qasm2_to_quafu_qc

        return qasm2_to_quafu_qc(self, openqasm)

    def to_openqasm(self, with_para=False) -> str:
        """
        Convert the circuit to openqasm text.

        Returns:
            openqasm text.
        """

        valid_gates = list(
            QuantumGate.gate_classes.keys()
        )  # TODO:include instruction futher
        valid_gates.extend(["barrier", "delay", "reset"])

        qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        qasm += "qreg q[%d];\n" % self.num
        qasm += "creg meas[%d];\n" % self.cbits_num
        if with_para:
            for variable in self.variables:
                qasm += f"{variable.latex} = {variable.value};\n"
        for gate in self.gates:
            if gate.name.lower() in valid_gates:
                qasm += gate.to_qasm(with_para) + ";\n"
            else:
                # TODO: add decompose subroutine
                raise NotImplementedError(
                    f"gate {gate.name} can not convert to qasm2 directly, you may decompose it manuallly"
                )
        for key in self.measures:
            qasm += "measure q[%d] -> meas[%d];\n" % (key, self.measures[key])

        self.openqasm = qasm
        return qasm

    def wrap_to_gate(self, name: str):
        """
        Wrap the circuit to a subclass of QuantumGate, create by metaclass.
        """
        from copy import deepcopy

        from quafu.elements.oracle import customize_gate

        # TODO: check validity of instructions
        gate_structure = [deepcopy(ins) for ins in self.instructions]
        customized = customize_gate(name, gate_structure, self.num)
        return customized

    def _reallocate(self, num, qbits: List[int]):
        """Remap the qubits and gates to new positions."""
        assert self.num == len(qbits)
        if max(qbits) > num:
            raise CircuitError("Bad allocation")

        self.num = num
        qbits_map = dict(zip(range(len(qbits)), qbits))
        gates = self.instructions
        for op in gates:
            for i in range(len(op.pos)):
                op.pos[i] = qbits_map[op.pos[i]]

            if isinstance(op, ControlledGate):
                for i in range(len(op.ctrls)):
                    op.ctrls[i] = qbits_map[op.ctrls[i]]

                for i in range(len(op.targs)):
                    op.targs[i] = qbits_map[op.targs[i]]

    def add_controls(
        self, ctrlnum, ctrls: List[int] = [], targs: List[int] = [], inplace=False
    ) -> "QuantumCircuit":
        num = 0
        instrs = []
        if len(ctrls + targs) == 0:
            ctrls = list(np.arange(ctrlnum) + self.num)
            num = self.num + ctrlnum
            instrs = self.instructions
        else:
            if len(ctrls) == 0 or len(targs) == 0:
                raise ValueError("Must provide both ctrls and targs")
            else:
                assert len(targs) == self.num
                assert len(ctrls) == ctrlnum
                num = max(ctrls + targs) + 1
                if inplace:
                    self._reallocate(num, targs)
                else:
                    temp = copy.deepcopy(self)
                    temp._reallocate(num, targs)
                    instrs = temp.instructions

        if inplace:
            for op in self.instructions:
                if isinstance(op, QuantumGate):
                    op = op.ctrl_by(ctrls)
            return self
        else:
            qc = QuantumCircuit(num)
            for op in instrs:
                if isinstance(op, QuantumGate):
                    qc << op.ctrl_by(ctrls)
        return qc

    def join(
        self,
        qc: ["QuantumCircuit", CircuitWrapper, ControlledCircuitWrapper],
        qbits: List[int] = [],
        inplace=True,
    ) -> "QuantumCircuit":
        num = self.num
        rnum = 0
        if isinstance(qc, QuantumCircuit):
            rnum = qc.num
        else:
            rnum = qc.circuit.num

        if len(qbits) == 0:
            num = max(self.num, rnum)
        else:
            num = max(self.num - 1, max(qbits)) + 1

        nqc = QuantumCircuit(num)
        if inplace:
            self.num = num
            nqc = self
        else:
            for op in self.instructions:
                nqc << op

        if isinstance(qc, QuantumCircuit):
            if qbits:
                newqc = copy.deepcopy(qc)
                newqc._reallocate(num, qbits)
                for op in newqc.instructions:
                    nqc << op
            else:
                for op in qc.instructions:
                    nqc << op
        else:
            if qbits:
                qc._reallocate(qbits)
            nqc << qc

        return nqc

    def power(self, n, inplace=False):
        if inplace:
            for _ in range(n - 1):
                self.instructions += self.instructions
            return self
        else:
            nq = QuantumCircuit(self.num)
            for _ in range(n):
                for op in self.instructions:
                    nq << op
            nq.measures = self.measures
            return nq

    def dagger(self, inplace=False):
        if inplace:
            self.instructions = self.instructions[::-1]
            for op in self.instructions:
                if isinstance(op, QuantumGate):
                    op = op.dagger()
            return self
        else:
            nq = QuantumCircuit(self.num)
            for op in self.instructions[::-1]:
                if isinstance(op, QuantumGate):
                    nq << op.dagger()
                else:
                    nq << op
            nq.measures = self.measures
            return nq

    def wrap(self, qbits=[]):
        # TODO:use OracleGate
        name = self.name if self.name else "Oracle"
        return CircuitWrapper(name, self, qbits)

    def unwrap(self):
        instructions = []
        gates = []
        for op in self.instructions:
            if isinstance(op, CircuitWrapper):
                circ = op.circuit.unwrap()
                for op_ in circ.instructions:
                    instructions.append(op_)
                    if isinstance(
                        op_, (QuantumGate, Delay, Barrier, XYResonance, KrausChannel)
                    ):
                        gates.append(op_)
            else:
                instructions.append(op)
                if isinstance(
                    op, (QuantumGate, Delay, Barrier, XYResonance, KrausChannel)
                ):
                    gates.append(op)

        self.instructions = instructions
        self._gates = gates
        self._has_wrap = False
        return self

    # # # # # # # # # # # # # # helper functions # # # # # # # # # # # # # #
    def id(self, pos: int) -> "QuantumCircuit":
        """
        Identity gate.

        Args:
            pos (int): qubit the gate act.
        """
        gate = qeg.IdGate(pos)
        self.add_ins(gate)
        return self

    def h(self, pos: int) -> "QuantumCircuit":
        """
        Hadamard gate.

        Args:
            pos (int): qubit the gate act.
        """
        gate = qeg.HGate(pos)
        self.add_ins(gate)
        return self

    def x(self, pos: int) -> "QuantumCircuit":
        """
        X gate.

        Args:
            pos (int): qubit the gate act.
        """
        gate = qeg.XGate(pos)
        self.add_ins(gate)
        return self

    def y(self, pos: int) -> "QuantumCircuit":
        """
        Y gate.

        Args:
            pos (int): qubit the gate act.
        """
        gate = qeg.YGate(pos)
        self.add_ins(gate)
        return self

    def z(self, pos: int) -> "QuantumCircuit":
        """
        Z gate.

        Args:
            pos (int): qubit the gate act.
        """
        self.add_ins(qeg.ZGate(pos))
        return self

    def t(self, pos: int) -> "QuantumCircuit":
        """
        T gate. (~Z^(1/4))

        Args:
            pos (int): qubit the gate act.
        """
        self.add_ins(qeg.TGate(pos))
        return self

    def tdg(self, pos: int) -> "QuantumCircuit":
        """
        Tdg gate. (Inverse of T gate)

        Args:
            pos (int): qubit the gate act.
        """
        self.add_ins(qeg.TdgGate(pos))
        return self

    def s(self, pos: int) -> "QuantumCircuit":
        """
        S gate. (~√Z)

        Args:
            pos (int): qubit the gate act.
        """
        self.add_ins(qeg.SGate(pos))
        return self

    def sdg(self, pos: int) -> "QuantumCircuit":
        """
        Sdg gate. (Inverse of S gate)

        Args:
            pos (int): qubit the gate act.
        """
        self.add_ins(qeg.SdgGate(pos))
        return self

    def sx(self, pos: int) -> "QuantumCircuit":
        """
        √X gate.

        Args:
            pos (int): qubit the gate act.
        """
        self.add_ins(qeg.SXGate(pos))
        return self

    def sxdg(self, pos: int) -> "QuantumCircuit":
        """
        Inverse of √X gate.

        Args:
            pos (int): qubit the gate act.
        """
        gate = qeg.SXdgGate(pos)
        self.add_ins(gate)
        return self

    def sy(self, pos: int) -> "QuantumCircuit":
        """
        √Y gate.

        Args:
            pos (int): qubit the gate act.
        """
        self.add_ins(qeg.SYGate(pos))
        return self

    def sydg(self, pos: int) -> "QuantumCircuit":
        """
        Inverse of √Y gate.

        Args:
            pos (int): qubit the gate act.
        """
        gate = qeg.SYdgGate(pos)
        self.add_ins(gate)
        return self

    def w(self, pos: int) -> "QuantumCircuit":
        """
        W gate. (~(X + Y)/√2)

        Args:
            pos (int): qubit the gate act.
        """
        self.add_ins(qeg.WGate(pos))
        return self

    def sw(self, pos: int) -> "QuantumCircuit":
        """
        √W gate.

        Args:
            pos (int): qubit the gate act.
        """
        self.add_ins(qeg.SWGate(pos))
        return self

    def rx(self, pos: int, para: ParameterType) -> "QuantumCircuit":
        """
        Single qubit rotation Rx gate.

        Args:
            pos (int): qubit the gate act.
            para (float): rotation angle
        """
        self.add_ins(qeg.RXGate(pos, para))
        return self

    def ry(self, pos: int, para: ParameterType) -> "QuantumCircuit":
        """
        Single qubit rotation Ry gate.

        Args:
            pos (int): qubit the gate act.
            para (float): rotation angle
        """
        self.add_ins(qeg.RYGate(pos, para))
        return self

    def rz(self, pos: int, para: ParameterType) -> "QuantumCircuit":
        """
        Single qubit rotation Rz gate.

        Args:
            pos (int): qubit the gate act.
            para (float): rotation angle
        """
        self.add_ins(qeg.RZGate(pos, para))
        return self

    def p(self, pos: int, para: ParameterType) -> "QuantumCircuit":
        """
        Phase gate

        Args:
            pos (int): qubit the gate act.
            para (float): rotation angle
        """
        self.add_ins(qeg.PhaseGate(pos, para))
        return self

    def cnot(self, ctrl: int, tar: int) -> "QuantumCircuit":
        """
        CNOT gate.

        Args:
            ctrl (int): control qubit.
            tar (int): target qubit.
        """
        self.add_ins(qeg.CXGate(ctrl, tar))
        return self

    def cx(self, ctrl: int, tar: int) -> "QuantumCircuit":
        """
        Ally of cnot.
        """
        return self.cnot(ctrl=ctrl, tar=tar)

    def cy(self, ctrl: int, tar: int) -> "QuantumCircuit":
        """
        Control-Y gate.

        Args:
            ctrl (int): control qubit.
            tar (int): target qubit.
        """
        self.add_ins(qeg.CYGate(ctrl, tar))
        return self

    def cz(self, ctrl: int, tar: int) -> "QuantumCircuit":
        """
        Control-Z gate.

        Args:
            ctrl (int): control qubit.
            tar (int): target qubit.
        """
        self.add_ins(qeg.CZGate(ctrl, tar))
        return self

    def cs(self, ctrl: int, tar: int) -> "QuantumCircuit":
        """
        Control-S gate.

        Args:
            ctrl (int): control qubit.
            tar (int): target qubit.
        """
        self.add_ins(qeg.CSGate(ctrl, tar))
        return self

    def ct(self, ctrl: int, tar: int) -> "QuantumCircuit":
        """
        Control-T gate.

        Args:
            ctrl (int): control qubit.
            tar (int): target qubit.
        """

        self.add_ins(qeg.CTGate(ctrl, tar))
        return self

    def cp(self, ctrl: int, tar: int, para: ParameterType) -> "QuantumCircuit":
        """
        Control-P gate.

        Args:
            ctrl (int): control qubit.
            tar (int): target qubit.
            para: theta
        """
        self.add_ins(qeg.CPGate(ctrl, tar, para))
        return self

    def swap(self, q1: int, q2: int) -> "QuantumCircuit":
        """
        SWAP gate

        Args:
            q1 (int): qubit the gate act.
            q2 (int): qubit the gate act.
        """
        self.add_ins(qeg.SwapGate(q1, q2))
        return self

    def iswap(self, q1: int, q2: int) -> "QuantumCircuit":
        """
        iSWAP gate

        Args:
            q1 (int): qubit the gate act.
            q2 (int): qubit the gate act.
        """
        self.add_ins(qeg.ISwapGate(q1, q2))
        return self

    def toffoli(self, ctrl1: int, ctrl2: int, targ: int) -> "QuantumCircuit":
        """
        Toffoli gate

        Args:
            ctrl1 (int): control qubit
            ctrl2 (int): control qubit
            targ (int): target qubit
        """
        self.add_ins(qeg.ToffoliGate(ctrl1, ctrl2, targ))
        return self

    def fredkin(self, ctrl: int, targ1: int, targ2: int) -> "QuantumCircuit":
        """
        Fredkin gate

        Args:
            ctrl (int):  control qubit
            targ1 (int): target qubit
            targ2 (int): target qubit
        """
        self.add_ins(qeg.FredkinGate(ctrl, targ1, targ2))
        return self

    def barrier(self, qlist: List[int] = None) -> "QuantumCircuit":
        """
        Add barrier for qubits in qlist.

        Args:
            qlist (list[int]): A list contain the qubit need add barrier. When qlist contain at least two qubit, the barrier will be added from minimum qubit to maximum qubit. For example: barrier([0, 2]) create barrier for qubits 0, 1, 2. To create discrete barrier, using barrier([0]), barrier([2]).
        """
        if qlist is None:
            qlist = list(range(self.num))
        self.add_ins(Barrier(qlist))
        return self

    def xy(self, qs: int, qe: int, duration: int, unit: str = "ns") -> "QuantumCircuit":
        """
        XY resonance time evolution for quantum simulator

        Args:
            qs: start position of resonant qubits.
            qe: end position of resonant qubits.
            duration: duration must be integer, which represents integer times of unit.
            unit: time unit of duration.

        """
        self.add_ins(XYResonance(qs, qe, duration, unit=unit))
        return self

    def rxx(self, q1: int, q2: int, theta):
        """
        Rotation about 2-qubit XX axis.

        Args:
            q1 (int): qubit the gate act.
            q2 (int): qubit the gate act.
            theta: rotation angle.

        """
        self.add_ins(qeg.RXXGate(q1, q2, theta))

    def ryy(self, q1: int, q2: int, theta):
        """
        Rotation about 2-qubit YY axis.

        Args:
            q1 (int): qubit the gate act.
            q2 (int): qubit the gate act.
            theta: rotation angle.

        """
        self.add_ins(qeg.RYYGate(q1, q2, theta))

    def rzz(self, q1: int, q2: int, theta):
        """
        Rotation about 2-qubit ZZ axis.

        Args:
            q1 (int): qubit the gate act.
            q2 (int): qubit the gate act.
            theta: rotation angle.

        """
        self.add_ins(qeg.RZZGate(q1, q2, theta))

    def mcx(self, ctrls: List[int], targ: int):
        """
        Multi-controlled X gate.

        Args:
            ctrls: A list of control qubits.
            targ: Target qubits.
        """
        self.add_ins(qeg.MCXGate(ctrls, targ))

    def mcy(self, ctrls: List[int], targ: int):
        """
        Multi-controlled Y gate.

        Args:
            ctrls: A list of control qubits.
            targ: Target qubits.
        """
        self.add_ins(qeg.MCYGate(ctrls, targ))

    def mcz(self, ctrls: List[int], targ: int):
        """
        Multi-controlled Z gate.

        Args:
            ctrls: A list of control qubits.
            targ: Target qubits.
        """
        self.add_ins(qeg.MCZGate(ctrls, targ))

    def unitary(self, matrix: np.ndarray, pos: List[int]):
        """
        Apply unitary to circuit on specified qubits.

        Args:
            matrix (np.ndarray): unitary matrix.
            pos (list[int]): qubits the gate act on.
        """
        compiler = UnitaryDecomposer(array=matrix, qubits=pos)
        compiler.apply_to_qc(self)

    def delay(self, pos, duration, unit="ns") -> "QuantumCircuit":
        """
        Let the qubit idle for a certain duration.

        Args:
            pos (int): qubit need delay.
            duration (int): duration of qubit delay, which represents integer times of unit.
            unit (str): time unit for the duration. Can be "ns" and "us".
        """
        self.add_ins(Delay(pos, duration, unit=unit))
        return self

    def reset(self, qlist: List[int] = None) -> "QuantumCircuit":
        """
        Add reset for qubits in qlist.

        Args:
            qlist (list[int]): A list contain the qubit need add reset. When qlist contain at least two qubit, the barrier will be added from minimum qubit to maximum qubit. For example: barrier([0, 2]) create barrier for qubits 0, 1, 2. To create discrete barrier, using barrier([0]), barrier([2]).

        Note: reset only support for simulator `qfvm_circ`.
        """
        if qlist is None:
            qlist = list(range(self.num))
        self.add_ins(Reset(qlist))
        self.executable_on_backend = False
        return self

    def measure(self, pos: List[int] = None, cbits: List[int] = None) -> None:
        """
        Measurement setting for experiment device.

        Args:
            pos: Qubits need measure.
            cbits: Classical bits keeping the measure results.
        """
        # checking
        if pos is None:
            pos = list(range(self.num))
        if np.any(np.array(pos) >= self.num):
            raise ValueError("Index out of range.")

        e_num = len(self.measures)  # existing num of measures
        n_num = len(pos)  # newly added num of measures
        if n_num > len(set(pos)):
            raise ValueError("Measured qubits not uniquely assigned.")

        if cbits:
            if not len(set(cbits)) == len(cbits):
                raise ValueError("Classical bits not uniquely assigned.")
            if not len(cbits) == n_num:
                raise ValueError(
                    "Number of measured bits should equal to the number of classical bits"
                )
        else:
            cbits = list(range(e_num, e_num + n_num))

        for cbit in cbits:
            if cbit < 0 or cbit > self.cbits_num:
                raise ValueError("Cbits index out of range.")
        # _sorted_indices = sorted(range(n_num), key=lambda k: cbits[k])
        # cbits = [_sorted_indices.index(i) + e_num for i in range(n_num)]

        measure = Measure(dict(zip(pos, cbits)))
        self._measures.append(measure)
        self.add_ins(measure)

    @contextmanager
    def cif(self, cbits: List[int], condition: int):
        """
        Create an `if` statement on this circuit.
        If cbits equals to condition, the subsequent operaterations will be performed.
        Use  the `measure` statement to explicitly assign value to the cbit before using it as `cbits` argument

        Args:
            cbits: List of cbit that are used for comparison.
            condition(int): A condition to be evaluated with cbits that filled by `measure` operation.


        For example::
            from quafu import QuantumCircuit
            qc = QuantumCircuit(2,2)

            qc.h(0)
            qc.cx(0,1)
            qc.measure([0],[0])
            with qc.cif(cbits=[0], condition=1):
                qc.x(2)
            qc.measure([2],[2])

        Note: cif only support for simulator `qfvm_circ`.
        """
        # check cbits
        if not len(set(cbits)) == len(cbits):
            raise ValueError("Classical bits not uniquely assigned.")
        if max(cbits) > self.cbits_num - 1 or min(cbits) < 0:
            raise ValueError("Classical bits index out of range.")
        # check condition
        if condition < 0:
            raise ValueError("Classical should be a positive integer.")
        self.executable_on_backend = False
        cif_ins = Cif(cbits=cbits, condition=condition)
        self.add_ins(cif_ins)

        yield

        instructions = []
        for i in range(len(self.instructions) - 1, -1, -1):
            if (
                isinstance(self.instructions[i], Cif)
                and self.instructions[i].instructions is None
            ):
                instructions.reverse()
                self.instructions[i].set_ins(instructions)
                self.instructions = self.instructions[0 : i + 1]
                return
            else:
                instructions.append(self.instructions[i])
