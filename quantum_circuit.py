

import numpy as np
import requests
import json
from urllib import parse
from collections import Iterable
from quantum_tools import ExecResult, merge_measure
from quantum_element import Barrier, QuantumGate
from element_gates import *

from transpiler.optsequence import OptSequence
# import pickle
# import jsonpickle

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
        self.result = []
        self.gates = []
        self.backend = "ScQ-P10"
        self.compiler = "default"
        self.qasm = []
        self.openqasm = ""
        self.gate_nodes = []
        self.optlist = []
        self.circuit = []
        self.measures = dict(zip(range(num), range(num)))
        self.qubits = list(range(num)) 

    def set_backend(self, backend):
        """
        Select the quantum device for executing task. Different computing backends provide different gate set. 

        Args:
            backend (str): "IOP" or "BAQIS".
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
        for gate in gatelist: 
            if isinstance(gate, SingleQubitGate) or isinstance(gate, ParaSingleQubitGate):
                gateQlist[gate.pos].append(gate)
            
            elif isinstance(gate, Barrier) or isinstance(gate, TwoQubitGate) or isinstance(gate, ParaTwoQubitGate):
                pos1 = min(gate.pos)
                pos2 = max(gate.pos)
                gateQlist[pos1].append(gate)
                for j in range(pos1+1, pos2+1):
                    gateQlist[j].append(None)

                maxlayer = max([len(gateQlist[j]) for j in gate.pos])
                for j in gate.pos:
                    layerj = len(gateQlist[j])
                    pos = layerj - 1
                    if not layerj == maxlayer:
                        for i in range(abs(layerj-maxlayer)):
                            gateQlist[j].insert(pos, None)

                
        maxdepth = max([len(gateQlist[i]) for i in range(num)])
        for gates in gateQlist:
            gates.extend([None] * (maxdepth-len(gates)))
        
        self.circuit = np.array(gateQlist)
        return self.circuit


    def draw_circuit(self):
        """
        Draw layered circuit using ASCII, print in terminal.
        """
        if len(self.circuit) == 0:
            self.layered_circuit()
        
        gateQlist = self.circuit
        num = gateQlist.shape[0]
        depth = gateQlist.shape[1]
        printlist = np.array([[""]*depth for i in range(2*num)], dtype="<U30")

        for l in range(depth):
            layergates = gateQlist[:, l]
            maxlen = 3
            for i in range(num):
                gate = layergates[i]
                if isinstance(gate, SingleQubitGate):
                    printlist[i*2, l] = gate.name
                    maxlen = max(maxlen, len(gate.name)+2)
                elif isinstance(gate, ParaSingleQubitGate):
                    gatestr = "%s(%.3f)" %(gate.name, gate.paras)
                    printlist[i*2, l] = gatestr
                    maxlen = max(maxlen, len(gatestr)+2)
                elif isinstance(gate, TwoQubitGate):
                    q1 = min(gate.pos)
                    q2 = max(gate.pos)
                    printlist[2*q1+1:2*q2, l] = "|"
                    if isinstance(gate, ControlGate):
                        printlist[gate.ctrl*2, l] = "*"
                        printlist[gate.targ*2, l] = "+"
                        
                        maxlen = max(maxlen, 5)
                        if gate.name != "CNOT":
                            printlist[q1+q2, l] = gate.name
                            maxlen = max(maxlen, len(gate.name)+2)
                    else:
                        if gate.name == "SWAP":
                            printlist[gate.pos[0]*2, l] = "*"
                            printlist[gate.pos[1]*2, l] = "*"
                        else: 
                            printlist[gate.pos[0]*2, l] = "#"
                            printlist[gate.pos[1]*2, l] = "#"
                            printlist[q1+q2, l] = gate.name
                            maxlen = max(maxlen, len(gate.name)+2)                
                elif isinstance(gate, ParaTwoQubitGate):
                    q1 = min(gate.pos)
                    q2 = max(gate.pos)
                    printlist[2*q1+1:2*q2, l] = "|"
                    printlist[gate.pos[0]*2, l] = "#"
                    printlist[gate.pos[1]*2, l] = "#"
                    gatestr = ""
                    if isinstance(gate.paras, Iterable):
                        gatestr = ("%s(" %gate.name + ",".join(["%.3f" %para for para in gate.paras]) +")")
                    else: gatestr = "%s(%.3f)" %(gate.name, gate.paras)
                    printlist[q1+q2, l] = gatestr
                    maxlen = max(maxlen, len(gatestr)+2)

                elif isinstance(gate, Barrier):
                    q1 = min(gate.pos)
                    q2 = max(gate.pos)
                    printlist[2*q1:2*q2+1, l] = "||"
                    maxlen = max(maxlen, len("||"))

            printlist[-1, l] = maxlen
        
        circuitstr = []
        for j in range(2*num-1):
            if j % 2 == 0:
                linestr = "%d " %(j//2) + "".join([printlist[j, l].center(int(printlist[-1, l]), "-") for l in range(depth)])
                if j//2 in self.measures.keys():
                    linestr += " M->c[%d]" %self.measures[j//2]
                circuitstr.append(linestr)
            else:
                circuitstr.append("  " + "".join([printlist[j, l].center(int(printlist[-1, l]), " ") for l in range(depth)]))
        circuitstr = "\n".join(circuitstr)
        print(circuitstr)

    def from_openqasm(self, openqasm):
        """
        Initialize the circuit from openqasm text.
        """
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
                self.num = int(qbs[2])
            elif operations == "creg":
                pass
            elif operations == "measure":
                mb = int(re.findall("\d+", operations_qbs[1])[0])
                cb = int(re.findall("\d+", operations_qbs[3])[0])
                self.measures[mb] = cb
            else:
                qbs = operations_qbs[1]    
                inds = [int(qb[2]) for qb in qbs.split(",")] 
                
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
                    elif gatename == "swap":
                        self.swap(inds[0], inds[1])
                    elif gatename == "iswap":
                        self.iswap(inds[0], inds[1])
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

        if not self.measures:
            self.measures = dict(zip(range(self.num), range(self.num)))

    def to_openqasm(self):
        """
        Convert the circuit to openqasm text.
        Returns: 
            openqasm text.
        """
        qasm = '''OPENQASM 2.0;\ninclude "qelib1.inc";\n'''
        qasm += "qreg q[%d];\n" %self.num
        qasm += "creg c[%d];\n" %len(self.measures)
        for gate in self.gates:
            if gate.name in "HXYZ":
                qasm += "%s q[%d];\n" %(gate.name.lower(), gate.pos)
            elif gate.name in ["Rx", "Ry", "Rz"]:
                qasm += "%s(%s) q[%d];\n" %(gate.name.lower(), gate.paras, gate.pos)
            elif gate.name == "CNOT":
                qasm += "cx q[%d],q[%d];\n" %(gate.ctrl, gate.targ)
            elif gate.name == "Cz":
                qasm += "cz q[%d],q[%d];\n" %(gate.ctrl, gate.targ)
            elif gate.name in ["SWAP", "iSWAP"]:
                qasm += "%s q[%d],q[%d];\n" %(gate.name.lower(), gate.pos[0], gate.pos[1])
            elif gate.name == "barrier":
                qasm += "barrier " + ",".join(["q[%d]" %p for p in gate.pos]) + ";\n"; 

        for key in self.measures:
            qasm += "measure q[%d] -> c[%d];\n" %(key, self.measures[key])

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
        #save input circuit
        inputs = self.gates

        if len(obslist) == 0:
            print("No observable measure task.")
            res = self.run()
            return res, [] 

        else:
            for obs in obslist:
                for p in obs[1]:
                    if p not in self.measures:
                        raise "Qubit %d in observer %s is not measured." %(p, obs[0])

           
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
                    self.ry(pos, -np.pi/2)
                elif base == "Y":
                    self.rx(pos, np.pi/2)
                                    
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
        data = {"qtasm": self.openqasm, "shots": self.shots, "qubits": self.num, "scan": 0, "tomo": int(self.tomo), "selected_server": backends[self.backend]}
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

    def x(self, pos):
        """
        X gate.

        Args:
            pos (int): qubit the gate act.
        """
        self.gates.append(XGate(pos))
    
    def y(self, pos):
        """
        Y gate.

        Args:
            pos (int): qubit the gate act.
        """
        self.gates.append(YGate(pos))
    
    def z(self, pos):
        """
        Z gate.

        Args:
            pos (int): qubit the gate act.
        """
        self.gates.append(ZGate(pos))
    
    def rx(self, pos, para):
        """
        Single qubit rotation Rx gate.

        Args:
            pos (int): qubit the gate act.
            para (float): rotation angle
        """
        if para != 0.:
            self.gates.append(RxGate(pos, para))
     
    def ry(self, pos, para):
        """
        Single qubit rotation Ry gate.
        
        Args:
            pos (int): qubit the gate act.
            para (float): rotation angle
        """
        if para != 0.:
            self.gates.append(RyGate(pos, para))
    
    def rz(self, pos, para):
        """
        Single qubit rotation Rz gate.
        
        Args:
            pos (int): qubit the gate act.
            para (float): rotation angle
        """
        if para != 0.:
            self.gates.append(RzGate(pos, para))
    
    def cnot(self, ctrl, tar):
        """
        CNOT gate.
        
        Args:
            ctrl (int): control qubit.
            tar (int): target qubit.
        """
        self.gates.append(CnotGate(ctrl, tar))
    
    def cz(self, ctrl, tar):
        """
        CZ gate.
        
        Args:
            ctrl (int): control qubit.
            tar (int): target qubit.
        """
        self.gates.append(CzGate(ctrl, tar))

    # def fsim(self, q1 ,q2, theta, phi):
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
        self.gates.append(SWAP([q1, q2]))

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

    def measure(self, pos, shots, cbits=[], tomo = False):
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
