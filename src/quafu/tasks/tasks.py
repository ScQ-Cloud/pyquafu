import os
from typing import Dict, List, Tuple

from quafu.circuits.quantum_circuit import QuantumCircuit
from ..utils.platform import get_homedir
from ..exceptions import CircuitError, ServerError, CompileError
from ..results.results import ExecResult, merge_measure
from ..backends.backends import ScQ_P10, ScQ_P20, ScQ_P50, ScQ_S41
from ..users.exceptions import UserError
import numpy as np
import json
import requests
from urllib import parse
import re
import copy
import networkx as nx
import matplotlib.pyplot as plt

class Task(object): 
    """
    Class for submitting quantum computation task to the backend.

    Attributes:
        token (str): Apitoken that associate to your Quafu account.
        shots (int): Numbers of single shot measurement.
        compile (bool): Whether compile the circuit on the backend
        tomo (bool): Whether do tomography (Not support yet)
    """
    def __init__(self):
        self._backend = ScQ_P10()
        self.token = ""
        self.shots = 1000
        self.tomo = False
        self.compile = True
        self._url = ""
        self.priority = 2
        self.submit_history = { }


    def load_account(self) -> None:
        """
        Load your Quafu account.
        """
        homedir = get_homedir()
        file_dir = homedir + "/.quafu/"
        try: 
            f = open(file_dir + "api", "r")
            data = f.readlines()
            self.token = data[0].strip("\n")
            self._url = data[1].strip("\n")
        except:
            raise UserError("User configure error. Please set up your token.")  

    def config(self, 
               backend: str="ScQ-P10", 
               shots: int=1000, 
               compile: bool=True,
               tomo: bool=False,
               priority: int=2) -> None:
        """
        Configure the task properties

        Args:
            backend: Select the experimental backend.
            shots: Numbers of single shot measurement.
            compile: Whether compile the circuit on the backend
            tomo:  Whether do tomography (Not support yet)
            priority: Task priority.
        """
        if backend == "ScQ-P10":
            self._backend = ScQ_P10()
        elif backend == "ScQ-P20":
            self._backend = ScQ_P20()
        elif backend == "ScQ-P50":
            self._backend = ScQ_P50()
        elif backend == "ScQ-S41":
            self._backend = ScQ_S41()        
        self.shots = shots
        self.tomo = tomo
        self.compile = compile
        self.priority = priority

    def get_history(self) -> Dict:
        """
        Get the history of submitted task.
        Returns:
            A dict of history. The key is the group name and the value is a list of task id in the group. 
        """
        return self.submit_history

    def get_backend_info(self) -> Dict:
        """
        Get the calibration information of the experimental backend.

        Returns: 
            Backend information dictionary containing the mapping from the indices to the names of physical bits `'mapping'`, backend topology  `'topology_diagram'` and full calibration inforamtion `'full_info'`.
        """
        backend_info = self._backend.get_info(self._url, self.token)
        json_topo_struct = backend_info["topological_structure"]
        qubits_list = []
        for gate in json_topo_struct.keys():
            qubit = gate.split('_')
            qubits_list.append(qubit[0])
            qubits_list.append(qubit[1])
        qubits_list = list(set(qubits_list))
        qubits_list = sorted(qubits_list, key=lambda x: int(re.findall(r"\d+", x)[0]))
        int_to_qubit = {k: v for k, v in enumerate(qubits_list)}
        qubit_to_int = {v: k for k, v in enumerate(qubits_list)}

        directed_weighted_edges = []
        weighted_edges = []
        edges_dict = {}
        clist = []
        for gate, name_fidelity in json_topo_struct.items():
            gate_qubit = gate.split('_')
            qubit1 = qubit_to_int[gate_qubit[0]]
            qubit2 = qubit_to_int[gate_qubit[1]]
            gate_name = list(name_fidelity.keys())[0]
            fidelity = name_fidelity[gate_name]['fidelity']
            directed_weighted_edges.append([qubit1, qubit2, fidelity])
            clist.append([qubit1, qubit2])
            gate_reverse = gate.split('_')[1] + '_' + gate.split('_')[0]
            if gate not in edges_dict and gate_reverse not in edges_dict:
                edges_dict[gate] = fidelity
            else:
                if fidelity < edges_dict[gate_reverse]:
                    edges_dict.pop(gate_reverse)
                    edges_dict[gate] = fidelity

        for gate, fidelity in edges_dict.items():
            gate_qubit = gate.split('_')
            qubit1, qubit2 = qubit_to_int[gate_qubit[0]], qubit_to_int[gate_qubit[1]]
            weighted_edges.append([qubit1, qubit2, np.round(fidelity, 3)])

        # draw topology

        G = nx.Graph()
        for key, value in int_to_qubit.items():
            G.add_node(key, name=value)

        G.add_weighted_edges_from(weighted_edges)

        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] >= 0.9]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] < 0.9]
        elarge_labels = {(u, v) : "%.3f" %d["weight"] for (u, v, d) in G.edges(data=True) if d["weight"] >= 0.9}
        esmall_labels = {(u, v) : "%.3f" %d["weight"] for (u, v, d) in G.edges(data=True) if d["weight"] < 0.9}

        pos = nx.spring_layout(G, seed=1)  
        fig, ax = plt.subplots()
        nx.draw_networkx_nodes(G, pos, node_size=400, ax=ax)

        nx.draw_networkx_edges(G, pos, edgelist=elarge, width=2, ax=ax)
        nx.draw_networkx_edges(
            G, pos, edgelist=esmall, width=2, alpha=0.5, style="dashed"
        , ax=ax)

        nx.draw_networkx_labels(G, pos, font_size=14, font_family="sans-serif", ax=ax)
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, elarge_labels, font_size=12, font_color="green", ax=ax)
        nx.draw_networkx_edge_labels(G, pos, esmall_labels, font_size=12, font_color="red", ax=ax)
        fig.set_figwidth(14)
        fig.set_figheight(14)
        fig.tight_layout()
        return {"mapping" : int_to_qubit, "topology_diagram": fig, "full_info": backend_info}

    def submit(self,
               qc: QuantumCircuit,
               obslist: List=[])\
                -> Tuple[List[ExecResult], List[int]]:
        """
        Execute the circuit with observable expectation measurement task.
        Args:
            qc (QuantumCircuit): Quantum circuit that need to be executed on backend.
            obslist (list[str, list[int]]): List of pauli string and its position.

        Returns: 
            List of executed results and list of measured observable

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
        inputs = copy.deepcopy(qc.gates)
        measures = list(qc.measures.keys())
        if len(obslist) == 0:
            print("No observable measurement task.")
            res = self.run(qc)
            return res, []

        else:
            for obs in obslist:
                for p in obs[1]:
                    if p not in measures:
                        raise CircuitError("Qubit %d in observer %s is not measured." % (p, obs[0]))

            measure_basis, targlist = merge_measure(obslist)
            print("Job start, need measured in ", measure_basis)

            exec_res = []
            for measure_base in measure_basis:
                res = self.run(qc, measure_base=measure_base)
                qc.gates = copy.deepcopy(inputs)
                exec_res.append(res)

            measure_results = []
            for obi in range(len(obslist)):
                obs = obslist[obi]
                rpos = [measures.index(p) for p in obs[1]]
                measure_results.append(exec_res[targlist[obi]].calculate_obs(rpos))

        return exec_res, measure_results

    def run(self, 
            qc: QuantumCircuit, 
            measure_base: List=[]) -> ExecResult:
        """Single run for measurement task.

        Args:
            qc (QuantumCircuit): Quantum circuit that need to be executed on backend.
            measure_base (list[str, list[int]]): measure base and it position.
        """
        if len(measure_base) == 0:
            res = self.send(qc)
            res.measure_base = ''

        else:
            for base, pos in zip(measure_base[0], measure_base[1]):
                if base == "X":
                    qc.ry(pos, -np.pi / 2)
                elif base == "Y":
                    qc.rx(pos, np.pi / 2)

            res = self.send(qc)
            res.measure_base = measure_base

        return res

    def send(self, 
             qc: QuantumCircuit, 
             name: str="", 
             group: str="",
            wait: bool=True) -> ExecResult:
        """
        Run the circuit on experimental device.

        Args:
            qc: Quantum circuit that need to be executed on backend.
            name: Task name.
            group: The task belong which group.
            wait: Whether wait until the execution return.
        Returns: 
            ExecResult object that contain the dict return from quantum device.
        """
        self.check_valid_gates(qc)
        qc.to_openqasm()
        backends = {"ScQ-P10": 0, "ScQ-P20": 1, "ScQ-P50": 2, "ScQ-S41" : 3}
        data = {"qtasm": qc.openqasm, "shots": self.shots, "qubits": qc.num, "scan": 0,
                "tomo": int(self.tomo), "selected_server": backends[self._backend.name],
                "compile": int(self.compile), "priority": self.priority, "task_name": name}
        
        if wait:
            url = self._url  + "qbackend/scq_kit/"
        else:
            url = self._url + "qbackend/scq_kit_asyc/"
            
        headers = {'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8', 'api_token': self.token}
        data = parse.urlencode(data)
        data = data.replace("%27", "'")
        res = requests.post(url, headers=headers, data=data)
        res_dict = json.loads(res.text)

        if res.json()["status"] in [201, 205]:
            raise UserError(res_dict["message"])
        elif res.json()["status"] == 5001:
            raise CircuitError(res_dict["message"])
        elif res.json()["status"] == 5003:
            raise ServerError(res_dict["message"])
        elif res.json()["status"] == 5004:
            raise CompileError(res_dict["message"]) 
        else:
            task_id = res_dict["task_id"]
     
            if not (group in self.submit_history):
                self.submit_history[group] = [task_id]
            else:
                self.submit_history[group].append(task_id)

            return ExecResult(res_dict, qc.measures)

    def retrieve(self, taskid: str) -> ExecResult:
        """
        Retrieve the results of submited task by taskid.
        
        Args:
            taskid: The taskid of the task need to be retrieved.
        """
        data = {"task_id" : taskid}
        url = self._url  + "qbackend/scq_task_recall/"

        headers = {'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8', 'api_token': self.token}
        res = requests.post(url, headers=headers, data=data)

        res_dict = json.loads(res.text)
        measures = eval(res_dict["measure"])
        
        return ExecResult(res_dict, measures)

    def retrieve_group(self, 
                       group: str,
                       history: Dict={}, 
                       verbose: bool=True) -> List[ExecResult]:
        """
        Retrieve the results of submited task by group name.

        Args:
            group: The group name need to be retrieved.
            history: History from which to retrieve the results. If not provided, the history will be the submit history of saved by current task.
            verbose: Whether print the task status in the group.
        Returns:
            A list of execution results in the retrieved group. Only completed task will be added. 
        """
        history = history if history else self.submit_history
        taskids = history[group]

        group_res = []
        if verbose:
            group = group if group else "Untitled group"
            print("Group: ", group)
            print((" "*5).join(["task_id".ljust(16), "task_name".ljust(10),   "status".ljust(10)]))
        for taskid in taskids:
            res = self.retrieve(taskid)
            taskname = res.taskname
            if verbose:
                taskname = taskname if taskname else "Untitled"
                print((" "*5).join([("%s" %res.taskid).ljust(16), ("%s" %taskname).ljust(10), ("%s" %res.task_status).ljust(10)]))
            if res.task_status == "Completed":
                group_res.append(res)

        return group_res
    
     
    def check_valid_gates(self, qc: QuantumCircuit) -> None:
        """
        Check the validity of the quantum circuit.
        Args:
            qc: QuantumCicuit object that need to be checked.
        """
        if not self.compile:
            valid_gates = self._backend.valid_gates
            for gate in qc.gates:
                if gate.name.lower() not in valid_gates:
                    raise CircuitError("Invalid operations '%s' for backend '%s'" %(gate.name, self._backend.name))
                    
        else:
            if self._backend.name == "ScQ-S41":
                raise CircuitError("Backend ScQ-S41 must be used without compilation")
            if self._backend.name == "ScQ-P50":
                for gate in qc.gates:
                    if gate.name.lower() in ["delay", "xy"]:
                        raise CircuitError("Invalid operations '%s' for backend '%s'" %(gate.name, self._backend.name))
        
