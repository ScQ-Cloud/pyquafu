import functools
import numpy as np
import requests
import json
from urllib import parse
# from collections import Iterable
from quantum_tools import ExecResult, merge_measure
from quantum_element import Barrier, QuantumGate, ControlGate, SingleQubitGate, TwoQubitGate
from element_gates import *
from typing import Iterable
from transpiler.optsequence import OptSequence
# import pickle
# import jsonpickle
import qutip


class QuantumCircuit(object):
    def __init__(self, num):
        """
        Initialize a QuantumCircuit object
        
        Args:
            num (int): Total qubit number used
        """
        self.num = num
        self.shots = 1000
        self.tomo = False
        self.gates = []
        self.backend = "ScQ-P10"
        self.openqasm = ""
        self.circuit = []
        self.measures = dict(zip(range(num), range(num)))
        self._compiled = False

    def set_backend(self, backend):
        """
        Select the quantum device for executing task. Different computing backends provide different gate set. 

        Args:
            backend (str): Can be "ScQ-P10", "ScQ-P20", "ScQ-P50".
        """
        self.backend = backend

    def get_backend(self):
        """
        Returns: 
            The backend used currently
        """
        return self.backend


    def layered_circuit(self):
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
            if isinstance(gate, SingleQubitGate):
                gateQlist[gate.pos].append(gate)
                if gate.pos not in used_qubits:
                    used_qubits.append(gate.pos)
            
            elif isinstance(gate, Barrier) or isinstance(gate, TwoQubitGate):
                pos1 = min(gate.pos)
                pos2 = max(gate.pos)
                gateQlist[pos1].append(gate)
                for j in range(pos1 + 1, pos2 + 1):
                    gateQlist[j].append(None)
              
                if isinstance(gate, TwoQubitGate):
                    for pos in gate.pos:
                        if pos not in used_qubits:
                            used_qubits.append(pos)

                maxlayer = max([len(gateQlist[j]) for j in range(pos1, pos2+1)])
                for j in range(pos1, pos2+1):
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
        return self.circuit


    def draw_circuit(self):
        """
        Draw layered circuit using ASCII, print in terminal.
        """
        self.layered_circuit()
        gateQlist = self.circuit
        num = gateQlist.shape[0]
        depth = gateQlist.shape[1] - 1
        printlist = np.array([[""] * depth for i in range(2 * num)], dtype="<U30")

        reduce_map = dict(zip(gateQlist[:, 0], range(num)))
        reduce_map_inv  = dict(zip(range(num), gateQlist[:, 0]))
        for l in range(depth):
            layergates = gateQlist[:, l+1]
            maxlen = 3
            for i in range(num):
                gate = layergates[i]
                if isinstance(gate, FixedSingleQubitGate):
                    printlist[i * 2, l] = gate.name
                    maxlen = max(maxlen, len(gate.name) + 2)
                elif isinstance(gate, ParaSingleQubitGate):
                    gatestr = "%s(%.3f)" % (gate.name, gate.paras)
                    printlist[i * 2, l] = gatestr
                    maxlen = max(maxlen, len(gatestr) + 2)
                elif isinstance(gate, FixedTwoQubitGate):
                    q1 = reduce_map[min(gate.pos)]
                    q2 = reduce_map[max(gate.pos)]
                    printlist[2 * q1 + 1:2 * q2, l] = "|"
                    if isinstance(gate, ControlGate):
                        printlist[reduce_map[gate.ctrl] * 2, l] = "*"
                        printlist[reduce_map[gate.targ] * 2, l] = "+"

                        maxlen = max(maxlen, 5)
                        if gate.name not in ["CNOT", "CX"] :
                            printlist[q1 + q2, l] = gate.name
                            maxlen = max(maxlen, len(gate.name) + 2)
                    else:
                        if gate.name == "SWAP":
                            printlist[reduce_map[gate.pos[0]] * 2, l] = "*"
                            printlist[reduce_map[gate.pos[1]] * 2, l] = "*"
                        else:
                            printlist[reduce_map[gate.pos[0]] * 2, l] = "#"
                            printlist[reduce_map[gate.pos[1]] * 2, l] = "#"
                            printlist[q1 + q2, l] = gate.name
                            maxlen = max(maxlen, len(gate.name) + 2)
                elif isinstance(gate, ParaTwoQubitGate):
                    q1 = reduce_map(min(gate.pos))
                    q2 = reduce_map(max(gate.pos))
                    printlist[2 * q1 + 1:2 * q2, l] = "|"
                    printlist[reduce_map[gate.pos[0]] * 2, l] = "#"
                    printlist[reduce_map[gate.pos[1]] * 2, l] = "#"
                    gatestr = ""
                    if isinstance(gate.paras, Iterable):
                        gatestr = ("%s(" % gate.name + ",".join(
                            ["%.3f" % para for para in gate.paras]) + ")")
                    else:
                        gatestr = "%s(%.3f)" % (gate.name, gate.paras)
                    printlist[q1 + q2, l] = gatestr
                    maxlen = max(maxlen, len(gatestr) + 2)

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
                linestr = ("q[%d]" %(reduce_map_inv[j//2])).ljust(6) + "".join([printlist[j, l].center(int(printlist[-1, l]), "-") for l in range(depth)])
                if reduce_map_inv[j//2] in self.measures.keys():
                    linestr += " M->c[%d]" %self.measures[reduce_map_inv[j//2]]
                circuitstr.append(linestr)
            else:
                circuitstr.append("".ljust(6) + "".join([printlist[j, l].center(int(printlist[-1, l]), " ") for l in range(depth)]))
        circuitstr = "\n".join(circuitstr)
        print(circuitstr)

    def from_openqasm(self, openqasm, compiled=False):
        """
        Initialize the circuit from openqasm text.
        """
        self._compiled = compiled
        from numpy import pi
        import re
        self.openqasm = openqasm
        lines = self.openqasm.splitlines()
        self.gates = []
        self.measures = {}
        for line in lines[2:]:
            operations_qbs = line.split(" ")
            operations = operations_qbs[0]
            if operations == "qreg":
                qbs = operations_qbs[1]
                self.num = int(re.findall("\d+", qbs)[0])
            elif operations == "creg":
                pass
            elif operations == "measure":
                mb = int(re.findall("\d+", operations_qbs[1])[0])
                cb = int(re.findall("\d+", operations_qbs[3])[0])
                self.measures[mb] = cb
            else:
                qbs = operations_qbs[1]    
                indstr = re.findall("\d+", qbs)
                inds = [int(indst) for indst in indstr]  
                
                if operations == "barrier":
                    self.barrier(inds)
                
                else:
                    sp_op = operations.split("(")
                    gatename = sp_op[0]
                    if len(sp_op) > 1:
                        paras = sp_op[1].strip("()")
                        parastr = paras.split(",")
                        paras = [eval(parai, {"pi":pi}) for parai in parastr]
                        
                    if gatename == "cx":
                        self.cnot(inds[0], inds[1])
                    elif gatename == "cy":
                        self.cy(inds[0], inds[1])
                    elif gatename == "cz":
                        self.cz(inds[0], inds[1])
                    elif gatename == "swap":
                        self.swap(inds[0], inds[1])
                    elif gatename == "rx":
                        self.rx(inds[0], paras[0])
                    elif gatename == "ry":
                        self.ry(inds[0], paras[0])
                    elif gatename == "rz":
                        self.ry(inds[0], paras[0])
                    elif gatename == "x":
                        self.x(inds[0])
                    elif gatename == "y":
                        self.y(inds[0])
                    elif gatename == "z":
                        self.z(inds[0])
                    elif gatename == "h":
                        self.h(inds[0])
                    elif gatename == "u1":
                        self.rz(inds[0], paras[0])
                    elif gatename == "u2":
                        self.rz(inds[0], paras[1])
                        self.ry(inds[0], pi/2)
                        self.rz(inds[0], paras[0])
                    elif gatename == "u3":
                        self.rz(inds[0], paras[2])
                        self.ry(inds[0], paras[0])
                        self.rz(inds[0], paras[1])
                    else:
                        print("operations %s may be not support currently" %gatename)

        if not self.measures:
            self.measures = dict(zip(range(self.num), range(self.num)))

    def to_openqasm(self):
        """
        Convert the circuit to openqasm text.
        Returns: 
            openqasm text.
        """
        qasm = '''OPENQASM 2.0;\ninclude "qelib1.inc";\n'''
        qasm += "qreg q[%d];\n" % self.num
        qasm += "creg meas[%d];\n" %len(self.measures)
        for gate in self.gates:
            if isinstance(gate, FixedSingleQubitGate):
                qasm += "%s q[%d];\n" % (gate.name.lower(), gate.pos)
            elif isinstance(gate, ParaSingleQubitGate):
                qasm += "%s(%s) q[%d];\n" % (gate.name.lower(), gate.paras, gate.pos)
            elif isinstance(gate, FixedTwoQubitGate):
                qasm += "%s q[%d],q[%d];\n" % (gate.name.lower(), gate.pos[0], gate.pos[1])
            elif type(gate) == Barrier:
                qasm += "barrier " + ",".join(["q[%d]" % p for p in gate.pos]) + ";\n"

        for key in self.measures:
            qasm += "measure q[%d] -> meas[%d];\n" %(key, self.measures[key])

        self.openqasm = qasm
        return qasm


    def submit_task(self, obslist=[]):
        """
        Execute the circuit with observable expectation measurement task.
        Args:
            obslist (list[str, list[int]]): List of pauli string and its position.

        Returns: 
            List of execut results and list of measured observable

        Examples: 
            1) input [["XYX", [0, 1, 2]], ["Z", [1]]] measure pauli operator XYX at 0, 1, 2 qubit, and Z at 1 qubit.\n
            2) Measure 5-qubit Ising Hamiltonian we can use\n
            obslist = [["X", [i]] for i in range(5)]]\n
            obslist.extend([["ZZ", [i, i+1]] for i in range(4)])\n
        
        For the energy expectation of Ising Hamiltonian \n
        res, obsexp = q.submit_task(obslist)\n
        E = sum(obsexp)
        """
        # save input circuit
        inputs = self.gates

        if len(obslist) == 0:
            print("No observable measure task.")
            res = self.run()
            return res, []

        else:
            for obs in obslist:
                for p in obs[1]:
                    if p not in self.measures:
                        raise "Qubit %d in observer %s is not measured." % (p, obs[0])

            measure_basis, targlist = merge_measure(obslist)
            print("Job start, need measured in ", measure_basis)

            exec_res = []
            for measure_base in measure_basis:
                res = self.run(measure_base=measure_base)
                self.gates = inputs
                exec_res.append(res)

            measure_results = []
            for obi in range(len(obslist)):
                obs = obslist[obi]
                rpos = [self.measures.index(p) for p in obs[1]]
                measure_results.append(exec_res[targlist[obi]].calculate_obs(rpos))

        return exec_res, measure_results

    def run(self, measure_base=[]):
        """Single run for measurement task.

        Args:
            measure_base (list[str, list[int]]): measure base and it position.
        """
        if len(measure_base) == 0:
            res = self.send()
            res.measure_base = ''

        else:
            for base, pos in zip(measure_base[0], measure_base[1]):
                if base == "X":
                    self.ry(pos, -np.pi / 2)
                elif base == "Y":
                    self.rx(pos, np.pi / 2)

            res = self.send()
            res.measure_base = measure_base

        return res

    def send(self):
        """
        Run the circuit on experimental device.

        Returns: 
            ExecResult object that contain the dict return from quantum device.
        """
        self.to_openqasm()
        backends = {"ScQ-P10":0, "ScQ-P20":1, "ScQ-P50":2}
        data = {"qtasm": self.openqasm, "shots": self.shots, "qubits": self.num, "scan": 0, "tomo": int(self.tomo), "selected_server": backends[self.backend], "compiled" : int(self._compiled)}
        url = "http://q.iphy.ac.cn/scq_submit_kit.php"
        headers = {'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'}
        data = parse.urlencode(data)
        data = data.replace("%27", "'")
        # data = data.replace("+", "")
        # data = data.replace("%20", " ")
        res = requests.post(url, headers = headers, data = data)

        if res.json()["stat"] == 5002:
            try:
                raise RuntimeError("Excessive computation scale.")
            except RuntimeError as e:
                print(e)

        elif res.json()["stat"] == 5001:
            try:
                raise RuntimeError("Invalid Circuit.")
            except RuntimeError as e:
                print(e)
        else:
            return ExecResult(json.loads(res.text))

    def h(self, pos):
        """
        Hadamard gate.

        Args:
            pos (int): qubit the gate act.
        """
        self.gates.append(HGate(pos))
        return self

    def x(self, pos):
        """
        X gate.

        Args:
            pos (int): qubit the gate act.
        """
        self.gates.append(XGate(pos))
        return self

    def y(self, pos):
        """
        Y gate.

        Args:
            pos (int): qubit the gate act.
        """
        self.gates.append(YGate(pos))
        return self

    def z(self, pos):
        """
        Z gate.

        Args:
            pos (int): qubit the gate act.
        """
        self.gates.append(ZGate(pos))
        return self

    def rx(self, pos, para):
        """
        Single qubit rotation Rx gate.

        Args:
            pos (int): qubit the gate act.
            para (float): rotation angle
        """
        if para != 0.:
            self.gates.append(RXGate(pos, para))
        return self

    def ry(self, pos, para):
        """
        Single qubit rotation Ry gate.
        
        Args:
            pos (int): qubit the gate act.
            para (float): rotation angle
        """
        if para != 0.:
            self.gates.append(RYGate(pos, para))
        return self

    def rz(self, pos, para):
        """
        Single qubit rotation Rz gate.
        
        Args:
            pos (int): qubit the gate act.
            para (float): rotation angle
        """
        if para != 0.:
            self.gates.append(RZGate(pos, para))
        return self

    def cnot(self, ctrl, tar):
        """
        CNOT gate.
        
        Args:
            ctrl (int): control qubit.
            tar (int): target qubit.
        """
        self.gates.append(CXGate([ctrl, tar]))
        return self

    def cy(self, ctrl, tar):
        """
        CY gate.

        Args:
            ctrl (int): control qubit.
            tar (int): target qubit.
        """
        self.gates.append(CYGate([ctrl, tar]))
        return self

    def cz(self, ctrl, tar):
        """
        CZ gate.
        
        Args:
            ctrl (int): control qubit.
            tar (int): target qubit.
        """
        self.gates.append(CZGate([ctrl, tar]))
        return self

    # def fsim(self, q1, q2, theta, phi):
    #     """
    #     fSim gate.
        
    #     Args:
    #         q1, q2 (int): qubits the gate act.
    #         theta (float): parameter theta in fSim. 
    #         phi (float): parameter phi in fSim.
    #     """
    #     self.gates.append(FsimGate([q1, q2], [theta, phi]))

    def swap(self, q1, q2):
        """
        SWAP gate
        
        Args:
            q1 (int): qubit the gate act.
            q2 (int): qubit the gate act.
        """
        self.gates.append(SwapGate([q1, q2]))
        return self
    # def iswap(self, q1, q2):
    #     """
    #     iSWAP gate

    #     Args:
    #         q1, q2 (int): qubits the gate act
    #     """
    #     self.gates.append(iSWAP([q1 ,q2]))

    def barrier(self, qlist):
        """
        Add barrier for qubits in qlist.
        
        Args:
            qlist (list[int]): A list contain the qubit need add barrier. When qlist contain at least two qubit, the barrier will be added from minimum qubit to maximum qubit. For example: barrier([0, 2]) create barrier for qubits 0, 1, 2. To create discrete barrier, using barrier([0]), barrier([2]).
        """
        self.gates.append(Barrier(qlist))
        return self

    def measure(self, pos, shots, cbits=[], tomo=False):
        """
        Measurement setting for experiment device.
        
        Args:
            pos (int): qubits need measure.
            shots (int): Sampling number for outcome state.
            tomo (bool): Whether do tomography.
        """

        self.measures = dict(zip(pos, range(len(pos))))
        self.shots = shots
        self.tomo = tomo
        if cbits:
            if len(cbits) ==  len(self.measures):
                self.measures = dict(zip(pos, cbits))
            else:
                raise ValueError("Number of measured bits should equal to the number of classical bits")

    def _operator(self):
        num = self.num
        assert num < 11
        oper = qutip.qeye([2] * num)
        for gate in self.gates:
            oper = gate._operator(num) * oper
        return oper

    def _simulate(self, result_type='prob'):
        num = self.num
        assert num < 11
        measures = list(self.measures.keys())
        psi = qutip.basis([2] * num, [0] * num)
        psi = self._operator() * psi
        rho = qutip.ptrace(psi, measures)
        if result_type.lower() in ['prob']:
            return np.abs(np.diag(rho)) ** 2
        elif result_type.lower() in ['tomo']:
            return np.array(rho)






