from typing import Iterable
import numpy as np
from ..elements.quantum_element import Barrier, Delay, ControlledGate, MultiQubitGate, ParaMultiQubitGate, QuantumGate, SingleQubitGate, XYResonance
from ..elements.element_gates import *
from ..exceptions import CircuitError

class QuantumCircuit(object):
    def __init__(self, num: int):
        """
        Initialize a QuantumCircuit object
        
        Args:   
            num (int): Total qubit number used
        """
        self.num = num
        self.gates = []
        self.openqasm = ""
        self.circuit = []
        self.measures = dict(zip(range(num), range(num)))
        self._used_qubits = []

    @property
    def used_qubits(self) -> List:
        self.layered_circuit()
        return self._used_qubits

    def add_gate(self, gate : QuantumGate):
        self.gates.append(gate)

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
            if isinstance(gate, SingleQubitGate) or isinstance(gate, Delay):
                gateQlist[gate.pos].append(gate)
                if gate.pos not in used_qubits:
                    used_qubits.append(gate.pos)

            elif isinstance(gate, Barrier) or isinstance(gate, MultiQubitGate) or isinstance(gate, XYResonance):
                pos1 = min(gate.pos)
                pos2 = max(gate.pos)
                gateQlist[pos1].append(gate)
                for j in range(pos1 + 1, pos2 + 1):
                    gateQlist[j].append(None)

                if isinstance(gate, MultiQubitGate) or isinstance(gate, XYResonance):
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

    def draw_circuit(self, width : int=4):
        """
        Draw layered circuit using ASCII, print in terminal.

        Args:
            width (int): The width of each gate. s
        """
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
                if isinstance(gate, SingleQubitGate) or isinstance(gate, Delay):
                    printlist[i * 2, l] = gate.symbol
                    maxlen = max(maxlen, len(gate.symbol) + width)

                elif isinstance(gate, MultiQubitGate) or isinstance(gate, XYResonance):
                    q1 = reduce_map[min(gate.pos)]
                    q2 = reduce_map[max(gate.pos)]
                    printlist[2 * q1 + 1:2 * q2, l] = "|"
                    printlist[q1 * 2, l] = "#"
                    printlist[q2 * 2, l] = "#"
                    if isinstance(gate, ControlledGate): #Controlled-Multiqubit gate
                        for ctrl in gate.ctrls:
                            printlist[reduce_map[ctrl] * 2, l] = "*"
                        
                        if gate.targ_name == "SWAP":
                            printlist[reduce_map[gate.targs[0]] * 2, l] = "x"
                            printlist[reduce_map[gate.targs[1]] * 2, l] = "x"
                        else:
                            tq1 = reduce_map[min(gate.targs)]
                            tq2 = reduce_map[max(gate.targs)]
                            printlist[tq1 * 2, l] = "#"
                            printlist[tq2 * 2, l] = "#"
                            if tq1 + tq2 in [reduce_map[ctrl] * 2 for ctrl in gate.ctrls]:
                                printlist[tq1 + tq2, l] = "*" + gate.symbol
                            else:
                                printlist[tq1 + tq2, l] = gate.symbol
                            maxlen = max(maxlen, len(gate.symbol) + width)
                                
                    else: #Multiqubit gate
                        if gate.name == "SWAP":
                            printlist[q1 * 2, l] = "x"
                            printlist[q2 * 2, l] = "x"

                        else:
                            printlist[q1 + q2, l] = gate.symbol
                            maxlen = max(maxlen, len(gate.symbol) + width)

                elif isinstance(gate, Barrier):
                    pos = [i for i in gate.pos if i in reduce_map.keys()]
                    q1 = reduce_map[min(pos)]
                    q2 = reduce_map[max(pos)]
                    printlist[2 * q1:2 * q2 + 1, l] = "||"
                    maxlen = max(maxlen, len("||"))
                

            printlist[-1, l] = maxlen

        circuitstr = []
        for j in range(2 * num - 1):
            if j % 2 == 0:
                linestr = ("q[%d]" % (reduce_map_inv[j // 2])).ljust(6) + "".join(
                    [printlist[j, l].center(int(printlist[-1, l]), "-") for l in range(depth)])
                if reduce_map_inv[j // 2] in self.measures.keys():
                    linestr += " M->c[%d]" % self.measures[reduce_map_inv[j // 2]]
                circuitstr.append(linestr)
            else:
                circuitstr.append("".ljust(6) + "".join(
                    [printlist[j, l].center(int(printlist[-1, l]), " ") for l in range(depth)]))
        circuitstr = "\n".join(circuitstr)
        print(circuitstr)
        return circuitstr

    def from_openqasm(self, openqasm : str):
        """
        Initialize the circuit from openqasm text.
        Args:
            openqasm: input openqasm str.
        """
        from numpy import pi
        import re
        self.openqasm = openqasm
        # lines = self.openqasm.strip("\n").splitlines(";")
        lines = self.openqasm.splitlines()
        lines = [line for line in lines if line]
        self.gates = []
        self.measures = {}
        measured_qubits = []
        global_valid = True
        for line in lines[2:]:
            if line:
                operations_qbs = line.split(" ", 1)
                operations = operations_qbs[0]
                if operations == "qreg":
                    qbs = operations_qbs[1]
                    self.num = int(re.findall("\d+", qbs)[0])
                elif operations == "creg":
                    pass
                elif operations == "measure":
                    qbs = operations_qbs[1]
                    indstr = re.findall("\d+", qbs)
                    inds = [int(indst) for indst in indstr]
                    mb = inds[0]
                    cb = inds[1]
                    self.measures[mb] = cb
                    measured_qubits.append(mb)
                else:
                    qbs = operations_qbs[1]
                    indstr = re.findall("\d+", qbs)
                    inds = [int(indst) for indst in indstr]
                    valid = True
                    for pos in inds:
                        if pos in measured_qubits:
                            valid = False
                            global_valid = False
                            break

                    if valid:
                        if operations == "barrier":
                            self.barrier(inds)

                        else:
                            sp_op = operations.split("(")
                            gatename = sp_op[0]
                            if gatename == "delay":
                                paras = sp_op[1].strip("()")
                                duration = int(re.findall("\d+", paras)[0])
                                unit = re.findall("[a-z]+", paras)[0]
                                self.delay(inds[0], duration, unit)
                            elif gatename == "xy":
                                paras = sp_op[1].strip("()")
                                duration = int(re.findall("\d+", paras)[0])
                                unit = re.findall("[a-z]+", paras)[0]
                                self.xy(min(inds), max(inds), duration, unit)
                            else:
                                if len(sp_op) > 1:
                                    paras = sp_op[1].strip("()")
                                    parastr = paras.split(",")
                                    paras = [eval(parai, {"pi": pi}) for parai in parastr]

                                if gatename == "cx":
                                    self.cnot(inds[0], inds[1])
                                elif gatename == "cy":
                                    self.cy(inds[0], inds[1])
                                elif gatename == "cz":
                                    self.cz(inds[0], inds[1])
                                elif gatename == "cp":
                                    self.cp(inds[0], inds[1], paras[0])
                                elif gatename == "swap":
                                    self.swap(inds[0], inds[1])
                                elif gatename == "rx":
                                    self.rx(inds[0], paras[0])
                                elif gatename == "ry":
                                    self.ry(inds[0], paras[0])
                                elif gatename == "rz":
                                    self.rz(inds[0], paras[0])
                                elif gatename == "p":
                                    self.p(inds[0], paras[0])
                                elif gatename == "x":
                                    self.x(inds[0])
                                elif gatename == "y":
                                    self.y(inds[0])
                                elif gatename == "z":
                                    self.z(inds[0])
                                elif gatename == "h":
                                    self.h(inds[0])
                                elif gatename == "id":
                                    self.id(inds[0])
                                elif gatename == "s":
                                    self.s(inds[0])
                                elif gatename == "sdg":
                                    self.sdg(inds[0])
                                elif gatename == "t":
                                    self.t(inds[0])
                                elif gatename == "tdg":
                                    self.tdg(inds[0])
                                elif gatename == "sx":
                                    self.sx(inds[0])
                                elif gatename == "ccx":
                                    self.toffoli(inds[0], inds[1], inds[2])
                                elif gatename == "cswap":
                                    self.fredkin(inds[0], inds[1], inds[2])
                                elif gatename == "u1":
                                    self.rz(inds[0], paras[0])
                                elif gatename == "u2":
                                    self.rz(inds[0], paras[1])
                                    self.ry(inds[0], pi / 2)
                                    self.rz(inds[0], paras[0])
                                elif gatename == "u3":
                                    self.rz(inds[0], paras[2])
                                    self.ry(inds[0], paras[0])
                                    self.rz(inds[0], paras[1])
                                elif gatename == "rxx":
                                    self.rxx(inds[0], inds[1], paras[0])
                                elif gatename == "ryy":
                                    self.ryy(inds[0], inds[1], paras[0])
                                elif gatename == "rzz":
                                    self.rzz(inds[0], inds[1], paras[0])
                                else:
                                    print(
                                        "Warning: Operations %s may be not supported by QuantumCircuit class currently." % gatename)

        if not self.measures:
            self.measures = dict(zip(range(self.num), range(self.num)))
        if not global_valid:
            print("Warning: All operations after measurement will be removed for executing on experiment")

    def to_openqasm(self) -> str:
        """
        Convert the circuit to openqasm text.

        Returns: 
            openqasm text.
        """
        qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n"
        qasm += "qreg q[%d];\n" % self.num
        qasm += "creg meas[%d];\n" % len(self.measures)
        for gate in self.gates:
            if isinstance(gate, SingleQubitGate):
                if isinstance(gate, ParaSingleQubitGate):
                    if isinstance(gate.paras, Iterable):
                        qasm += "%s(" %gate.name.lower() + ",".join(["%s" %para for para in gate.paras]) + ") q[%d];\n" % (gate.pos)
                    else:
                        qasm += "%s(%s) " %(gate.name.lower(), gate.paras) + "q[%d];\n" % (gate.pos)
                else:
                    if gate.name == "SY":
                        qasm += "ry(pi/2) q[%d];\n" %(gate.pos)
                    elif gate.name == "W":
                        qasm += "rz(-pi/4) q[%d];\nrx(pi) q[%d];\nrz(pi/4) q[%d];\n"  %(gate.pos, gate.pos, gate.pos)
                    elif gate.name == "SW":
                        qasm += "rz(-pi/4) q[%d];\nrx(pi/2) q[%d];\nrz(pi/4) q[%d];\n"  %(gate.pos, gate.pos, gate.pos)
                    else:
                        qasm += "%s q[%d];\n" % (gate.name.lower(), gate.pos)

            elif isinstance(gate, Delay):
                qasm += "delay(%d%s) q[%d];\n" % (gate.duration, gate.unit, gate.pos)
            elif isinstance(gate, XYResonance):
                qasm += "xy(%d%s) " %(gate.duration, gate.unit) + ",".join(["q[%d]" % p for p in range(min(gate.pos), max(gate.pos)+1)]) + ";\n"
                
            elif isinstance(gate, Barrier) or isinstance(gate, MultiQubitGate):
                if isinstance(gate, ParaMultiQubitGate) or (isinstance(gate, ControlledGate) and bool(gate.paras)):
                    if isinstance(gate.paras, Iterable):
                        qasm += "%s(" %gate.name.lower() + ",".join(["%s" %para for para in gate.paras]) + ") " + ",".join(["q[%d]" % p for p in gate.pos]) + ";\n"
                    else:
                         qasm += "%s(%s) " %(gate.name.lower(), gate.paras) + ",".join(["q[%d]" % p for p in gate.pos]) + ";\n"
                
                else:
                    if gate.name == "CS":
                        qasm += "cp(pi/2) " + "q[%d],q[%d];\n" % (gate.pos[0], gate.pos[1])
                    elif gate.name == "CT":
                        qasm += "cp(pi/4) " + "q[%d],q[%d];\n" % (gate.pos[0], gate.pos[1])
                    elif gate.name == "barrier":
                        qasm += "barrier " + ",".join(["q[%d]" % p for p in range(min(gate.pos), max(gate.pos)+1)]) + ";\n"
                    else:
                        qasm += "%s " %(gate.name.lower()) + ",".join(["q[%d]" % p for p in gate.pos]) + ";\n"

        for key in self.measures:
            qasm += "measure q[%d] -> meas[%d];\n" % (key, self.measures[key])

        self.openqasm = qasm
        return qasm


    def id(self, pos: int) -> "QuantumCircuit":
        """
        Identity gate.

        Args:
            pos (int): qubit the gate act.
        """ 
        self.gates.append(IdGate(pos))
        return self

    def h(self, pos: int) -> "QuantumCircuit":
        """
        Hadamard gate.

        Args:
            pos (int): qubit the gate act.
        """
        self.gates.append(HGate(pos))
        return self

    def x(self, pos: int) -> "QuantumCircuit":
        """
        X gate.

        Args:
            pos (int): qubit the gate act.
        """
        self.gates.append(XGate(pos))
        return self

    def y(self, pos: int) -> "QuantumCircuit":
        """
        Y gate.

        Args:
            pos (int): qubit the gate act.
        """
        self.gates.append(YGate(pos))
        return self

    def z(self, pos: int) -> "QuantumCircuit":
        """
        Z gate.

        Args:
            pos (int): qubit the gate act.
        """
        self.gates.append(ZGate(pos))
        return self

    def t(self, pos: int) -> "QuantumCircuit":
        """
        T gate. (~Z^(1/4))

        Args:
            pos (int): qubit the gate act.
        """
        self.gates.append(TGate(pos))
        return self
    
    def tdg(self, pos: int) -> "QuantumCircuit":
        """
        Tdg gate. (Inverse of T gate)

        Args:
            pos (int): qubit the gate act.
        """
        self.gates.append(TdgGate(pos))

    def s(self, pos: int) -> "QuantumCircuit":
        """
        S gate. (~√Z)

        Args:
            pos (int): qubit the gate act.
        """
        self.gates.append(SGate(pos))
        return self

    def sdg(self, pos: int) -> "QuantumCircuit":
        """
        Sdg gate. (Inverse of S gate)

        Args:
            pos (int): qubit the gate act.
        """
        self.gates.append(SdgGate(pos))
        return self

    def sx(self, pos: int) -> "QuantumCircuit":
        """
        √X gate.

        Args:
            pos (int): qubit the gate act.
        """
        self.gates.append(SXGate(pos))
        return self

    def sy(self, pos: int) -> "QuantumCircuit":
        """
        √Y gate.

        Args:
            pos (int): qubit the gate act.
        """
        self.gates.append(SYGate(pos))
        return self

    def w(self, pos: int) -> "QuantumCircuit":
        """
        W gate. (~(X + Y)/√2)

        Args:
            pos (int): qubit the gate act.
        """
        self.gates.append(WGate(pos))
        return self
    
    def sw(self, pos: int) -> "QuantumCircuit":
        """
        √W gate.

        Args:
            pos (int): qubit the gate act.
        """
        self.gates.append(SWGate(pos))
        return self
    
    def rx(self, pos: int, para: float) -> "QuantumCircuit":
        """
        Single qubit rotation Rx gate.

        Args:
            pos (int): qubit the gate act.
            para (float): rotation angle
        """
        self.gates.append(RXGate(pos, para))
        return self

    def ry(self, pos: int, para: float) -> "QuantumCircuit":
        """
        Single qubit rotation Ry gate.
        
        Args:
            pos (int): qubit the gate act.
            para (float): rotation angle
        """
        self.gates.append(RYGate(pos, para))
        return self

    def rz(self, pos: int, para: float) -> "QuantumCircuit":
        """
        Single qubit rotation Rz gate.
        
        Args:
            pos (int): qubit the gate act.
            para (float): rotation angle
        """
        self.gates.append(RZGate(pos, para))
        return self

    def p(self, pos: int, para: float) -> "QuantumCircuit":
        """
        Phase gate
        
        Args:
            pos (int): qubit the gate act.
            para (float): rotation angle
        """
        self.gates.append(PhaseGate(pos, para))

    def cnot(self, ctrl: int, tar: int) -> "QuantumCircuit":
        """
        CNOT gate.
        
        Args:
            ctrl (int): control qubit.
            tar (int): target qubit.
        """
        self.gates.append(CXGate(ctrl, tar))
        return self

    def cy(self, ctrl: int, tar: int) -> "QuantumCircuit":
        """
        Control-Y gate.

        Args:
            ctrl (int): control qubit.
            tar (int): target qubit.
        """
        self.gates.append(CYGate(ctrl, tar))
        return self

    def cz(self, ctrl: int, tar: int) -> "QuantumCircuit":
        """
        Control-Z gate.
        
        Args:
            ctrl (int): control qubit.
            tar (int): target qubit.
        """
        self.gates.append(CZGate(ctrl, tar))
        return self

    def cs(self, ctrl: int, tar: int) -> "QuantumCircuit":
        """
        Control-S gate.
        Args:
            ctrl (int): control qubit.
            tar (int): target qubit.
        """
        self.gates.append(CSGate(ctrl, tar))
        return self

    def ct(self, ctrl: int, tar: int) -> "QuantumCircuit":
        """
        Control-T gate.
        Args:
            ctrl (int): control qubit.
            tar (int): target qubit.
        """
        
        self.gates.append(CTGate(ctrl, tar))
        return self

    def cp(self, ctrl: int, tar: int, para) -> "QuantumCircuit":
        """
        Control-P gate.

        Args:
            ctrl (int): control qubit.
            tar (int): target qubit.
        """
        self.gates.append(CPGate(ctrl, tar, para))
        return self
  

    def swap(self, q1: int, q2: int) -> "QuantumCircuit":
        """
        SWAP gate
        
        Args:
            q1 (int): qubit the gate act.
            q2 (int): qubit the gate act.
        """
        self.gates.append(SwapGate(q1, q2))
        return self

    def toffoli(self, ctrl1: int, ctrl2: int, targ: int) -> "QuantumCircuit":
        """
        Toffoli gate

        Args:
            ctrl1 (int): control qubit
            ctrl2 (int): control qubit
            targ (int): target qubit
        """
        self.gates.append(ToffoliGate(ctrl1, ctrl2, targ))
        return self
    
    def fredkin(self, ctrl: int, targ1:int , targ2: int) -> "QuantumCircuit":
        """
        Fredkin gate
        
        Args:
            ctrl (int):  control qubit
            targ1 (int): target qubit
            targ2 (int): target qubit
        """
        self.gates.append(FredkinGate(ctrl, targ1, targ2))
        return self

    def barrier(self, qlist: List[int]) -> "QuantumCircuit":
        """
        Add barrier for qubits in qlist.
        
        Args:
            qlist (list[int]): A list contain the qubit need add barrier. When qlist contain at least two qubit, the barrier will be added from minimum qubit to maximum qubit. For example: barrier([0, 2]) create barrier for qubits 0, 1, 2. To create discrete barrier, using barrier([0]), barrier([2]).
        """
        self.gates.append(Barrier(qlist))
        return self

    def delay(self, pos, duration, unit="ns") -> "QuantumCircuit":
        """
        Let the qubit idle for a certain duration.

        Args:
            pos (int): qubit need delay.
            duration (int): duration of qubit delay, which represents integer times of unit.
            unit (str): time unit for the duration. Can be "ns" and "us". 
        """
        self.gates.append(Delay(pos, duration, unit=unit))
        return self

    def xy(self, qs: int, qe: int, duration: int, unit: str="ns") -> "QuantumCircuit":
        """
        XY resonance time evolution for quantum simulator
        Args:
            qs: start position of resonant qubits.
            qe: end position of resonant qubits.
            duration: duration must be integer, which represents integer times of unit.
            unit: time unit of duration.

        """
        self.gates.append(XYResonance(qs, qe, duration, unit=unit))
        return self

    def rxx(self, q1: int, q2: int, theta):
        """
        Rotation about 2-qubit XX axis.
        Args:
            q1 (int): qubit the gate act.
            q2 (int): qubit the gate act.
            theta: rotation angle.

        """
        self.gates.append(RXXGate(q1, q2, theta))
    
    def ryy(self, q1: int, q2: int, theta):
        """
        Rotation about 2-qubit YY axis.
        Args:
            q1 (int): qubit the gate act.
            q2 (int): qubit the gate act.
            theta: rotation angle.

        """
        self.gates.append(RYYGate(q1, q2, theta))

    def rzz(self, q1: int, q2: int, theta):
        """
        Rotation about 2-qubit ZZ axis.
        Args:
            q1 (int): qubit the gate act.
            q2 (int): qubit the gate act.
            theta: rotation angle.

        """
        self.gates.append(RZZGate(q1, q2, theta))
    
    def mcx(self, ctrls: List[int], targ: int):
        """
        Multi-controlled X gate.
        """
        self.gates.append(MCXGate(ctrls, targ))
    
    def mcy(self, ctrls: List[int], targ: int):
        """
        Multi-controlled X gate.
        """
        self.gates.append(MCYGate(ctrls, targ))
    
    def mcz(self, ctrls: List[int], targ: int):
        """
        Multi-controlled X gate.
        """
        self.gates.append(MCZGate(ctrls, targ))
    

    def measure(self, pos: List[int], cbits: List[int] = []) -> None:
        """
        Measurement setting for experiment device.
        
        Args:
            pos: Qubits need measure.
            cbits: Classical bits keeping the measure results.
        """

        self.measures = dict(zip(pos, range(len(pos))))

        if cbits:
            if len(cbits) == len(self.measures):
                self.measures = dict(zip(pos, cbits))
            else:
                raise CircuitError("Number of measured bits should equal to the number of classical bits")



