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
import logging
from typing import Dict, List, Optional, Tuple
from urllib import parse

import numpy as np
from quafu.circuits.quantum_circuit import QuantumCircuit
from quafu.users.userapi import User

from ..exceptions import CircuitError, UserError, validate_server_resp
from ..results.results import ExecResult, merge_measure
from ..utils.client_wrapper import ClientWrapper


class Task:
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

    def __init__(self, user: Optional[User] = None):
        self.user = User() if user is None else user

        self.shots = 1000
        self.tomo = False
        self.compile = False
        self.priority = self.user.priority
        self.runtime_job_id = ""
        self.submit_history = {}
        self._available_backends = self.user.get_available_backends(print_info=False)
        self.backend = self._available_backends[
            list(self._available_backends.keys())[0]
        ]

    def config(
        self,
        backend: str = "ScQ-P10",
        shots: int = 1000,
        compile: bool = True,
        tomo: bool = False,
        priority: int = 2,
    ) -> None:
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
            raise UserError(
                "backend %s is not valid, available backends are " % backend
                + ", ".join(self._available_backends.keys())
            )

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

    def submit(
        self, qc: QuantumCircuit, obslist: List = []
    ) -> Tuple[List[ExecResult], List[int]]:
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
                        raise CircuitError(
                            "Qubit %d in observer %s is not measured." % (p, obs[0])
                        )

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

    def run(self, qc: QuantumCircuit, measure_base: List = None) -> ExecResult:
        """Single run for measurement task.

        Args:
            qc (QuantumCircuit): Quantum circuit that need to be executed on backend.
            measure_base (list[str, list[int]]): measure base and its positions.
        """
        if measure_base is None:
            res = self.send(qc)
            res.measure_base = ""

        else:
            for base, pos in zip(measure_base[0], measure_base[1]):
                if base == "X":
                    qc.ry(pos, -np.pi / 2)
                elif base == "Y":
                    qc.rx(pos, np.pi / 2)

            res = self.send(qc)
            res.measure_base = measure_base

        return res

    def send(
        self, qc: QuantumCircuit, name: str = "", group: str = "", wait: bool = False
    ) -> ExecResult:
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
            raise CircuitError(
                "The qubit number %d is too large for backend %s which has %d qubits"
                % (qc.num, self.backend.name, self.backend.qubit_num)
            )

        if self.backend.name not in ["ScQ-P156", "ScQ-P106"]:
            self.check_valid_gates(qc)
        qc.to_openqasm()
        data = {
            "qtasm": qc.openqasm,
            "shots": self.shots,
            "qubits": qc.num,
            "scan": 0,
            "tomo": int(self.tomo),
            "selected_server": self.backend.system_id,
            "compile": int(self.compile),
            "priority": self.priority,
            "task_name": name,
            "pyquafu_version": version,
            "runtime_job_id": self.runtime_job_id,
        }

        if wait:
            url = User.url + User.exec_api
        else:
            url = User.url + User.exec_async_api

        logging.debug("quantum circuit validated, sending task...")
        headers = {
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            "api_token": self.user.api_token,
        }
        data = parse.urlencode(data)
        data = data.replace("%27", "'")
        response = ClientWrapper.post(
            url, headers=headers, data=data
        )  # type: requests.models.Response

        # TODO: completing status code checks
        # FIXME: Maybe we need to delete below code
        if not response.ok:
            logging.warning("Received a non-200 response from the server.\n")
        if response.status_code == 502:
            logging.critical(
                "Received a 502 Bad Gateway response. Please try again later.\n"
                "If there is persistent failure, please report it on our github page."
            )
            raise UserError("502 Bad Gateway response")
        # FIXME: Maybe we need to delete above code

        res_dict = response.json()  # type: dict
        validate_server_resp(res_dict)

        task_id = res_dict["task_id"]

        if group not in self.submit_history:
            self.submit_history[group] = [task_id]
        else:
            self.submit_history[group].append(task_id)

        return ExecResult(res_dict)

    def retrieve(self, taskid: str) -> ExecResult:
        """
        Retrieve the results of submited task by taskid.

        Args:
            taskid: The taskid of the task need to be retrieved.
        """
        data = {"task_id": taskid}
        url = User.url + User.exec_recall_api

        headers = {
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            "api_token": self.user.api_token,
        }
        response = ClientWrapper.post(url, headers=headers, data=data)

        res_dict = response.json()
        return ExecResult(res_dict)

    def retrieve_group(
        self, group: str, history: Dict = None, verbose: bool = True
    ) -> List[ExecResult]:
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
            print(
                (" " * 5).join(
                    ["task_id".ljust(16), "task_name".ljust(10), "status".ljust(10)]
                )
            )
        for taskid in taskids:
            res = self.retrieve(taskid)
            taskname = res.taskname
            if verbose:
                taskname = taskname if taskname else "Untitled"
                print(
                    (" " * 5).join(
                        [
                            ("%s" % res.taskid).ljust(16),
                            ("%s" % taskname).ljust(10),
                            ("%s" % res.task_status).ljust(10),
                        ]
                    )
                )
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
                    raise CircuitError(
                        "Invalid operations '%s' for backend '%s'"
                        % (gate.name, self.backend.name)
                    )

        else:
            # TODO: use system_id for ScQ-S41
            if self.backend.name == "ScQ-S41":
                raise CircuitError("Backend ScQ-S41 must be used without compilation")
            if self.backend.system_id == 2:  # ScQ-P136
                for gate in qc.gates:
                    if gate.name.lower() in ["xy"]:
                        raise CircuitError(
                            "Invalid operations '%s' for backend '%s'"
                            % (gate.name, self.backend.name)
                        )
