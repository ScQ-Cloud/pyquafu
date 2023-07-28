import copy
import json
from typing import Dict, List, Tuple
from urllib import parse

import numpy as np
import requests

from quafu.circuits.quantum_circuit import QuantumCircuit
from quafu.users.userapi import User
from ..exceptions import CircuitError, ServerError, CompileError
from ..results.results import ExecResult, merge_measure
from ..users.exceptions import UserError


class Task(object):
    """
    Class for submitting quantum computation task to the backend.

    Attributes:
        shots (int): Numbers of single shot measurement.
        compile (bool): Whether compile the circuit on the backend
        tomo (bool): Whether do tomography (Not support yet)
        user (User): User object corresponding to Quafu account
        priority (int): priority level of the task
        submit_history (dict): circuit submitted with this task
        backend (dict): quantum backend that execute the task.

    """

    def __init__(self, user=User()):
        # update api-token, a patch to be deleted in the future
        user._api_token = user._load_account_token()
        self.user = user

        self.shots = 1000
        self.tomo = False
        self.compile = True
        self.priority = self.user.priority
        self.runtime_job_id = ""
        self.submit_history = {}
        self._available_backends = self.user.get_available_backends(print_info=False)
        self.backend = self._available_backends[list(self._available_backends.keys())[0]]

    def config(self,
               backend: str = "ScQ-P10",
               shots: int = 1000,
               compile: bool = True,
               tomo: bool = False,
               priority: int = 2) -> None:
        """
        Configure the task properties

        Args:
            backend: Select the experimental backend.
            shots: Numbers of single shot measurement.
            compile: Whether compile the circuit on the backend
            tomo:  Whether to do tomography (Not support yet)
            priority: Task priority.
        """
        if backend not in self._available_backends.keys():
            raise UserError("backend %s is not valid, available backends are " % backend + ", ".join(
                self._available_backends.keys()))

        self.backend = self._available_backends[backend]
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
        return self.backend.get_chip_info(self.user)

    def submit(self,
               qc: QuantumCircuit,
               obslist: List = []) \
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
            measure_base: List = []) -> ExecResult:
        """Single run for measurement task.

        Args:
            qc (QuantumCircuit): Quantum circuit that need to be executed on backend.
            measure_base (list[str, list[int]]): measure base and its positions.
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
             name: str = "",
             group: str = "",
             wait: bool = True) -> ExecResult:
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
        from quafu import get_version
        version = get_version()
        if qc.num > self.backend.qubit_num:
            raise CircuitError("The qubit number %d is too large for backend %s which has %d qubits" % (
            qc.num, self.backend.name, self.backend.qubit_num))

        self.check_valid_gates(qc)
        qc.to_openqasm()
        data = {"qtasm": qc.openqasm, "shots": self.shots, "qubits": qc.num, "scan": 0,
                "tomo": int(self.tomo), "selected_server": self.backend.system_id,
                "compile": int(self.compile), "priority": self.priority, "task_name": name,
                "pyquafu_version": version, "runtime_job_id": self.runtime_job_id}

        if wait:
            url = User.exec_api
        else:
            url = User.exec_async_api

        headers = {'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8', 'api_token': self.user.api_token}
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
        data = {"task_id": taskid}
        url = User.exec_recall_api

        headers = {'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8', 'api_token': self.user.api_token}
        res = requests.post(url, headers=headers, data=data)

        res_dict = json.loads(res.text)
        measures = eval(res_dict["measure"])

        return ExecResult(res_dict, measures)

    def retrieve_group(self,
                       group: str,
                       history: Dict = {},
                       verbose: bool = True) -> List[ExecResult]:
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
            print((" " * 5).join(["task_id".ljust(16), "task_name".ljust(10), "status".ljust(10)]))
        for taskid in taskids:
            res = self.retrieve(taskid)
            taskname = res.taskname
            if verbose:
                taskname = taskname if taskname else "Untitled"
                print((" " * 5).join(
                    [("%s" % res.taskid).ljust(16), ("%s" % taskname).ljust(10), ("%s" % res.task_status).ljust(10)]))
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
            valid_gates = self.backend.get_valid_gates()
            for gate in qc.gates:
                if gate.name.lower() not in valid_gates:
                    raise CircuitError("Invalid operations '%s' for backend '%s'" % (gate.name, self.backend.name))

        else:
            if self.backend.name == "ScQ-S41":
                raise CircuitError("Backend ScQ-S41 must be used without compilation")
            if self.backend.name == "ScQ-P136":
                for gate in qc.gates:
                    if gate.name.lower() in ["xy"]:
                        raise CircuitError("Invalid operations '%s' for backend '%s'" % (gate.name, self.backend.name))
