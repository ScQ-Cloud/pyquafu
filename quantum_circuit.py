

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
        self.measures = [] 
        self.shots = 1000
        self.tomo = False
        self.result = []
        self.gates = []
        self.backend = "IOP"
        self.compiler = "default"
        self.qasm = []
        self.openqasm = ""
        self.gate_nodes = []
        self.optlist = []
        self.circuit = []
        self.qubits = list(range(num)) #default physical qubits
        self.p2v = dict(zip(range(num), range(num))) #{physical:virtual}
        self.v2p = dict(zip(range(num), range(num))) #{virtual:physical}
        self.trivial_mapping = True
        self.cbits = []

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

    # def set_compiler(self, compiler="default"):
    #     """
    #     Choose the compiler that convert circuit to qasm.

    #     Args:
    #         compiler (str): Complier used to generate QASM conde. 
    #         values: 
             
    #          "optseq": Left just the circuit and make the layout valid for experiment device and optimized the totoal execution time. For certain layout or large scale circuit, this compiler is time cost. Considering using "default" and ajust the circuit to valid forms manually.

    #          "default": Only left justed layout. You may add Barrier manually to make the circuit vaild for experiment device. You should avoid any gate
    #             adjoint to a two-qubit gate in one layer. 

    #     Example: 
    #             q = QuantumCircuit(5)\n
    #             q.x(0)\n
    #             q.cnot(1, 2)\n
    #             is not valid, you can use\n
    #             q = QuantumCircuit(5)\n
    #             q.barrier([0])\n
    #             q.x(0)\n
    #             q.cnot(1, 2)\n
    #             This will make CNOT gate in the first layer while X gate in the second layer, which is valid for experimental device.
    #         """
    #     self.compiler = compiler
    
    def check_mapping(self):
        self.trival_mapping = True
        for (key, values) in self.p2v:
            if key != values:
                self.trival_mapping = False
                break

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

    # def from_qasm(self, qasm):
    #     """
    #     Generate layered circuit from input qasm text.

    #     Args:
    #             qasm (str): Raw qasm text for quantum device. Now only support the IOP form 
    #     """
    #     if self.backend == "IOP":
    #         layer_list = eval(qasm)
    #         depth = len(layer_list) - 2
    #         gateQlist = np.empty((self.num, depth), dtype=QuantumGate)
    #         gateQlist.fill(None)
    #         for j in range(depth):
    #             layer_gates = layer_list[j]
    #             for layer_gate in layer_gates:
    #                 if layer_gate[0] == "X":
    #                     gateQlist[layer_gate[1], j] = XGate(layer_gate[1])
    #                 elif layer_gate[0] == "Y":
    #                     gateQlist[layer_gate[1], j] = YGate(layer_gate[1])
    #                 elif layer_gate[0] == "Z":
    #                     gateQlist[layer_gate[1], j] = ZGate(layer_gate[1])
    #                 elif layer_gate[0] == "H":
    #                     gateQlist[layer_gate[1], j] = HGate(layer_gate[1])
    #                 elif layer_gate[0] == "Rx":
    #                     gateQlist[layer_gate[1], j] = RxGate(layer_gate[1], layer_gate[2])
    #                 elif layer_gate[0] == "Ry":
    #                     gateQlist[layer_gate[1], j] = RyGate(layer_gate[1], layer_gate[2])
    #                 elif layer_gate[0] == "Rz":
    #                     gateQlist[layer_gate[1], j] = RzGate(layer_gate[1], layer_gate[2])
    #                 elif layer_gate[0] == "CNOT":
    #                     gateQlist[layer_gate[1][0], j] = CnotGate(layer_gate[1][0], layer_gate[1][1])
    #                 elif layer_gate[0] == "Cz":
    #                     gateQlist[layer_gate[1][0], j] = CzGate(layer_gate[1][0], layer_gate[1][1])
    #                 elif layer_gate[0] == "iSWAP":
    #                     gateQlist[layer_gate[1][0], j] = iSWAP([layer_gate[1][0], layer_gate[1][1]])
    #                 elif layer_gate[0] == "SWAP":
    #                     gateQlist[layer_gate[1][0], j] = iSWAP([layer_gate[1][0], layer_gate[1][1]])

    #         self.circuit = gateQlist
    #         self.measures = layer_list[-2]
    #         self.qubits = layer_list[-1]

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
                circuitstr.append("%d " %self.p2v[j//2] + "".join([printlist[j, l].center(int(printlist[-1, l]), "-") for l in range(depth)]))
            else:
                circuitstr.append("  " + "".join([printlist[j, l].center(int(printlist[-1, l]), " ") for l in range(depth)]))
        circuitstr = "\n".join(circuitstr)
        print(circuitstr)

    def from_openqasm(self, openqasm):
        self.openqasm = openqasm
        lines = self.openqasm.splitlines()
        from numpy import pi
        self.gates = []
        for line in lines[2:]:
            operations_qbs = line.split(" ")
            operations = operations_qbs[0]
            if operations == "qreg":
                qbs = operations_qbs[1]
                self.num = int(qbs[2])
            elif operations == "creg":
                pass
            elif operations == "measure":
                pass
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
                    

    def to_openqasm(self):
        qasm = '''OPENQASM 2.0;\ninclude "qelib1.inc";\n'''
        qasm += "qreg q[%d];\n" %self.num
        # qasm += "creg meas[%d];\n" %len(self.measures)
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

        # for j in range(len(self.measures)):
        #     qasm += "measure q[%d] -> meas[%d];\n" %(self.measures[j], self.cbits[j])

        self.openqasm = qasm
        return qasm
    
   
    # def compile_to_qLisp(self):
    #     """
    #     Compile the circuit to QASM that excute on BAQIS backend
    #     """
    #     qasm_qLisp = []
    #     for gate in self.gates:
    #         qasm_qLisp.append(gate.to_QLisp())

    #     if not len(self.measures) == 0:
    #         for measure_bits in self.measures:
    #             qasm_qLisp.append((("Measure", measure_bits), "Q%d" %measure_bits))
    #     else:
    #         raise "No qubit measured"

    #     self.qasm = qasm_qLisp
    

    # def compile_to_IOP(self):
    #     """
    #     Compile the circuit to QASM that execute on IOP backend

    #     """
    #     if self.compiler == "optseq":
    #         self.gate_nodes = []
    #         for gate in self.gates:
    #             if not isinstance(gate, Barrier):
    #                 self.gate_nodes.append(gate.to_nodes())
    #         opt = OptSequence(self.num, self.gate_nodes)
    #         opt.initial()
    #         opt.run_opt(set(opt.zerodeg), 0, opt.deg, [])
    #         self.optlist = opt.optlist
    #         allstrs = []
    #         for i in self.optlist:
    #             m = "["
    #             templist = []
    #             for j in i:
    #                 templist.append(j)
    #             m += ','.join(templist)
    #             m += "]"
    #             allstrs.append(m)
    #         totstr = ",".join(allstrs)
    #         totstr += ",["
    #         templist = []
    #         for i in self.measures:
    #             templist.append("%d" % i)
    #         totstr += ",".join(templist)
    #         templist = []
    #         totstr += "],["
    #         for i in list(range(self.num)):
    #             templist.append("%d" % i)
    #         totstr += ",".join(templist)
    #         totstr += "]"
    #         qasm_IOP = totstr
    #         self.qasm = qasm_IOP

    #     elif self.compiler == "default":
    #         gateQlist = self.layered_circuit()
    #         totalgates = []
    #         for l in range(gateQlist.shape[1]):
    #             layergate = [gate.to_IOP() for gate in gateQlist[:, l] if gate != None and not isinstance(gate, Barrier)]
    #             if len(layergate) != 0:
    #                 totalgates.append(layergate)
    #         measures_p = [self.v2p[i] for i in self.measures]
    #         totalgates.append(measures_p)
    #         totalgates.append(self.qubits)
    #         self.qasm = str(totalgates)[1:-1]

    def submit_task(self, obslist=[]):
        """
        Execute the circuit with observable expectation measurement task.
        Args:
            obslist list(str, list[int]): List of pauli string and its position.

        Returns: 
            List of execut results and list of measured observable

        Example: 
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
            measure_base (list(str, list[int])) measure base and it position.
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

        data = {"qtasm": self.openqasm, "shots": self.shots, "qubits": self.num, "scan": 0, "tomo": int(self.tomo), "selected_server": 0}
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

    def fsim(self, q1 ,q2, theta, phi):
        """
        fSim gate.
        
        Args:
            q1, q2 (int): qubits the gate act.
            theta (float): parameter theta in fSim. 
            phi (float): parameter phi in fSim.
        """
        self.gates.append(FsimGate([q1, q2], [theta, phi]))
    
    def swap(self, q1, q2):
        """
        SWAP gate
        
        Args:
            q1, q2 (int): qubits the gate act
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
            qlist (list(int)): A list contain the qubit need add barrier. When qlist contain at least two qubit, the barrier will be added from minimum qubit to maximum qubit. For example: barrier([0, 2]) create barrier for qubits 0, 1, 2. To create discrete barrier, using\n
            barrier([0])\n
            barrier([2]) \n
        """
        self.gates.append(Barrier(qlist))

    def measure(self, pos, shots, cbits=[], tomo = False):
        """
        Measurement setting for experiment device.
        
        Args:
            pos (int): qubits need measure.
            shot (int): Sampling number for outcome state.
            tomo (bool): Whether do tomography.
        """

        self.measures = pos
        self.shots = shots
        self.tomo = tomo
        if cbits:
            if len(cbits) ==  self.measures:
                self.cbits = cbits
            else:
                raise ("Number of measured bits should equal to the number of classical bit")
        else:
            self.cbits = list(range(len(self.measures)))
    