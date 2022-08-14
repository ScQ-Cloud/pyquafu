from operator import ge
import os
from ..utils.platform import get_homedir
from ..exceptions import CircuitError, ServerError, CompileError
from ..results.results import ExecResult, merge_measure
from ..backends.backends import ScQ_P10, ScQ_P20, ScQ_P50
from ..users.exceptions import UserError
import numpy as np
import json
import requests
from urllib import parse

class Task(object):
    """
    Class for submitting quantum computation task to the backend.
    Attributes:
        token (str) : Apitoken that associate to your Quafu account.
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
    
    def load_account(self):
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

    def config(self, backend="ScQ_P10", shots=1000, compile=True, tomo=False):
        """
        Configure the task properties
        Args:
            backend (str): Select the experimental backend.
            shots (int): Numbers of single shot measurement.
            compile (bool): Whether compile the circuit on the backend
            tomo (bool): Whether do tomography (Not support yet)
        """
        if backend == "ScQ-P10":
            self._backend = ScQ_P10()
        elif backend == "ScQ-P20":
            self._backend = ScQ_P20()
        elif backend == "ScQ-P50":
            self._backend = ScQ_P50()
        
        self.shots = shots
        self.tomo = tomo
        self.compile = compile

    def get_backend_info(self):
        return self._backend.get_info(self._url, self.token)

    def submit(self, qc, obslist=[]):
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
        inputs = qc.gates
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
                qc.gates = inputs
                exec_res.append(res)

            measure_results = []
            for obi in range(len(obslist)):
                obs = obslist[obi]
                rpos = [measures.index(p) for p in obs[1]]
                measure_results.append(exec_res[targlist[obi]].calculate_obs(rpos))

        return exec_res, measure_results

    def run(self, qc, measure_base=[]):
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

    def send(self, qc):
        """
        Run the circuit on experimental device.

        Args:
            qc (QuantumCircuit): Quantum circuit that need to be executed on backend.
        Returns: 
            ExecResult object that contain the dict return from quantum device.
        """
        qc.to_openqasm()
        backends = {"ScQ-P10": 0, "ScQ-P20": 1, "ScQ-P50": 2}
        data = {"qtasm": qc.openqasm, "shots": self.shots, "qubits": qc.num, "scan": 0,
                "tomo": int(self.tomo), "selected_server": backends[self._backend.name],
                "compile": int(self.compile)}
        print(data)
        url = self._url  + "qbackend/scq_kit/"
        headers = {'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8', 'api_token': self.token}
        data = parse.urlencode(data)
        data = data.replace("%27", "'")
        res = requests.post(url, headers=headers, data=data)
        res_dict = json.loads(res.text)

        if res.json()["status"] == 201:
            raise UserError(res_dict["message"])
        elif res.json()["status"] == 5001:
            raise CircuitError(res_dict["message"])
        elif res.json()["status"] == 5003:
            raise ServerError(res_dict["message"])
        elif res.json()["status"] == 5004:
            raise CompileError(res_dict["message"]) 
        else: 
            return ExecResult(res_dict, qc.measures)

