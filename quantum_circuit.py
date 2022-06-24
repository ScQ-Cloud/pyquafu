
import requests
import json
from urllib import parse
import numpy as np
from quantum_tools import *
from element_gates import *
from transpiler import OptSequence
# import pickle
# import jsonpickle

class QuantumCircuit(object):
    def __init__(self, num):
        self.num = num        
        self.measures = [] 
        self.shots = 1000
        self.tomo = False
        self.result = []
        self.gates = []
        self.backend = "IOP"
        self.qasm = []
        self.gate_nodes = []
        self.optlist = []

    def set_backend(self, backend):
        self.backend = backend

    def get_backend(self, backend):
        return self.backend

    def layered_circuit(self):
        num = self.num
        gatelist = self.gates
        gateQlist = [[] for i in range(num)]
        for gate in gatelist: 
            if isinstance(gate, SingleQubitGate) or isinstance(gate, ParaSingleQubitGate):
                gateQlist[gate.pos].append(gate)
            
            elif isinstance(gate, Barrier) or isinstance(gate, TwoQubitGate) or isinstance(gate, ParaTwoQubitGate):
                gateQlist[gate.pos[0]].append(gate)
                for j in gate.pos[1:]:
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
        
        return np.array(gateQlist) 


    def draw_circuit(self):
        gateQlist = self.layered_circuit()
        num = gateQlist.shape[0]
        depth = gateQlist.shape[1]
        printlist = np.array([[""]*depth for i in range(2*num)], dtype="<U30")

        for l in range(depth):
            layergates = gateQlist[:, l]
            maxlen = 5
            for i in range(num):
                gate = layergates[i]
                if isinstance(gate, SingleQubitGate):
                    printlist[i*2, l] = gate.name
                    maxlen = max(maxlen, len(gate.name)+4)
                elif isinstance(gate, ParaSingleQubitGate):
                    gatestr = "%s(%.3f)" %(gate.name, gate.paras)
                    printlist[i*2, l] = gatestr
                    maxlen = max(maxlen, len(gatestr)+4)
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
                            maxlen = max(maxlen, len(gate.name)+4)
                    else:
                        if gate.name == "SWAP":
                            printlist[gate.pos[0]*2, l] = "*"
                            printlist[gate.pos[1]*2, l] = "*"
                        else: 
                            printlist[gate.pos[0]*2, l] = "#"
                            printlist[gate.pos[1]*2, l] = "#"
                            printlist[q1+q2, l] = gate.name
                            maxlen = max(maxlen, len(gate.name)+4)                
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
                    maxlen = max(maxlen, len(gatestr)+4)

                elif isinstance(gate, Barrier):
                    q1 = min(gate.pos)
                    q2 = max(gate.pos)
                    printlist[2*q1:2*q2+1, l] = "||"
                    maxlen = max(maxlen, len("||"))

            printlist[-1, l] = maxlen
        
        circuitstr = []
        for j in range(2*num-1):
            if j % 2 == 0:
                circuitstr.append("".join([printlist[j, l].center(int(printlist[-1, l]), "-") for l in range(depth)]))
            else:
                circuitstr.append("".join([printlist[j, l].center(int(printlist[-1, l]), " ") for l in range(depth)]))
        circuitstr = "\n".join(circuitstr)
        print(circuitstr)

    def compile_to_qLisp(self):
        qasm_qLisp = []
        for gate in self.gates:
            qasm_qLisp.append(gate.to_QLisp())

        if not len(self.measures) == 0:
            for measure_bits in self.measures:
                qasm_qLisp.append((("Measure", measure_bits), "Q%d" %measure_bits))
        else:
            raise "No qubit measured"

        self.qasm = qasm_qLisp
        

    def compile_to_IOP(self, compiler="optseq"):
        if compiler == "optseq":
            self.gate_nodes = []
            for gate in self.gates:
                self.gate_nodes.append(gate.to_nodes())
            opt = OptSequence(self.num, self.gate_nodes)
            opt.initial()
            opt.run_opt(set(opt.zerodeg), 0, opt.deg, [])
            self.optlist = opt.optlist
            allstrs = []
            for i in self.optlist:
                m = "["
                templist = []
                for j in i:
                    templist.append(j)
                m += ','.join(templist)
                m += "]"
                allstrs.append(m)
            totstr = ",".join(allstrs)
            totstr += ",["
            templist = []
            for i in self.measures:
                templist.append("%d" % i)
            totstr += ",".join(templist)
            templist = []
            totstr += "],["
            for i in list(range(self.num)):
                templist.append("%d" % i)
            totstr += ",".join(templist)
            totstr += "]"
            qasm_IOP = totstr
            self.qasm = qasm_IOP
        elif compiler == "qcover":
            pass

        elif compiler == "default":
            gateQlist = self.layered_circuit()
            totalgates = []
            for l in range(gateQlist.shape[1]):
                layergate = [gate.to_IOP() for gate in gateQlist[:, l] if gate != None and not isinstance(gate, Barrier)]
                totalgates.append(layergate)
            totalgates.append(list(self.measures))
            totalgates.append(list(range(self.num)))
            self.qasm = str(totalgates)[1:-1]

    def submit_task(self, psi, obslist=[]):
        #save input circuit
        inputs = self.gates

        if len(obslist) == 0:
            res = self.run(psi)
            return res 

        else:
            for obs in obslist:
                for p in obs[1]:
                    if p not in self.measures:
                        raise "Qubit %d in observer %s is not measured." %(p, obs[0])

           
            measure_basis, targlist = merge_measure(obslist)
            print("Job start, need measured in ", measure_basis)

            #TODO: Send the measure_basis and use the returned exec_res
            # for measure_base in measure_basis:
            #     measure_base[0] = list(measure_base)

            exec_res = []
            for measure_base in measure_basis:
                res = self.run(psi, measure_base=measure_base)
                self.gates = inputs
                exec_res.append(res)


            measure_results = []
            for obi in range(len(obslist)):
                obs = obslist[obi]
                rpos = [self.measures.index(p) for p in obs[1]]
                measure_results.append(exec_res[targlist[obi]].calculate_obs(rpos))

        return exec_res, measure_results

    def run(self, psi, measure_base=[]):
        if len(measure_base) == 0:
            res = self.send()
            res['measure_base'] = ''

        else:
            for base, pos in zip(measure_base[0], measure_base[1]):
                if base == "X":
                    self.ry(pos, -np.pi/2)
                elif base == "Y":
                    self.rx(pos, np.pi/2)
                                    
            res = self.send()
            res['measure_base'] = measure_base

        return ExecResult(res)

    def send(self, compiler="optseq"):
        if self.backend == "IOP":
            self.compile_to_IOP(compiler = compiler)
        elif self.backend == "BAQIS":
            self.compile_to_qLisp()
        else:
            raise "Wrong backend"

            # print(self.qasm)
        data = {"qtasm": self.qasm, "shots": self.shots, "qubits": 10, "scan": 0, "tomo": int(self.tomo), "selected_server": 0}
        url = "http://q.iphy.ac.cn/scq_submit_task.php"
        headers = {'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'}
        data = parse.urlencode(data)
        data = data.replace("%27", "'")
        data = data.replace("+", "")
        # print(self.qasm)
        res = requests.post(url, headers = headers, data = data)
        # print(res.json())

        if res.json()["stat"] == 5002:
            try:
                raise RuntimeError("Excessive computation scale.")
            except RuntimeError as e:
                print(e)

        elif res.json()["stat"] == 5001:
            try:
                raise RuntimeError("Invalid Circuit")
            except RuntimeError as e:
                print(e)
        else:
            return ExecResult(json.loads(res.text))


    def h(self, pos):
        self.gates.append(HGate(pos))

    def x(self, pos):
        self.gates.append(XGate(pos))
    
    def y(self, pos):
        self.gates.append(YGate(pos))
    
    def z(self, pos):
        self.gates.append(ZGate(pos))
    
    def rx(self, pos, para):
        self.gates.append(RxGate(pos, para))
     
    def ry(self, pos, para):
        self.gates.append(RyGate(pos, para))
    
    def rz(self, pos, para):
        self.gates.append(RzGate(pos, para))
    
    def cnot(self, ctrl, tar):
        self.gates.append(CnotGate(ctrl, tar))
    
    def cz(self, ctrl, tar):
        self.gates.append(CzGate(ctrl, tar))

    def fsim(self, q1 ,q2, theta, phi):
        self.gates.append(FsimGate([q1, q2], [theta, phi]))
    
    def iswap(self, q1, q2):
        self.gates.append(iSWAP([q1 ,q2]))
     
    def barrier(self, qlist):
        self.gates.append(Barrier(qlist))

    def measure(self, pos, shots, tomo = False):
        self.measures = pos
        self.shots = shots
        self.tomo = tomo
    