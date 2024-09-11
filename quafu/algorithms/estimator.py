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
"""Pre-build wrapper to calculate expectation value"""
import copy
import time
from typing import List, Optional

import numpy as np
from quafu.exceptions.quafu_error import CircuitError
from quafu.results.results import ExecResult, merge_measure

from ..circuits.quantum_circuit import QuantumCircuit
from ..simulators import simulate
from ..tasks.tasks import Task
from .hamiltonian import Hamiltonian


def execute_circuit(circ: QuantumCircuit, observables: Hamiltonian):
    """Execute circuit on quafu simulator"""
    sim_res = simulate(circ, hamiltonian=observables)
    expectations = sim_res["pauli_expects"]
    return sum(expectations)


# TODO: cache measure results values and reuse for expectation calculation
class Estimator:
    """Estimate expectation for quantum circuits and observables"""

    def __init__(
        self,
        circ: QuantumCircuit,
        backend: str = "sim",
        task: Optional[Task] = None,
        **task_options
    ) -> None:
        """
        Args:
            circ: quantum circuit.
            backend: run on simulator (sim) or real machines (ScQ-PXX)
            task: task instance for real machine execution (should be none if backend is "sim")
            task_options: options to config a task instance
        """
        self._circ = circ
        self._circ.get_parameter_grads()  # parameter shift currently requires calling this for initialization
        self._backend = backend
        self._task = None
        if backend != "sim":
            if task is not None:
                self._task = task
            else:
                self._task = Task()
            self._task.config(backend=self._backend, **task_options)

    def _run_real_machine(self, observables: Hamiltonian):
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
        if not isinstance(self._task, Task):
            raise ValueError("_task not initiated in Estimator")
        # TODO(zhaoyilun):
        #   investigate the best implementation for calculating
        #   expectation on real devices.
        obslist = observables.to_pauli_list()

        # save input circuit
        inputs = copy.deepcopy(self._circ.gates)
        measures = list(self._circ.measures.keys())
        if len(obslist) == 0:
            print("No observable measurement task.")
            res = self._measure_obs(self._circ)
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
            lst_task_id = []
            for measure_base in measure_basis:
                res = self._measure_obs(self._circ, measure_base=measure_base)
                self._circ.gates = copy.deepcopy(inputs)
                lst_task_id.append(res.taskid)

            for tid in lst_task_id:
                # retrieve task results
                while True:
                    res = self._task.retrieve(tid)
                    if res.task_status == "Completed":
                        exec_res.append(res)
                        break
                    time.sleep(0.2)

            measure_results = []
            for obi in range(len(obslist)):
                obs = obslist[obi]
                rpos = [measures.index(p) for p in obs[1]]
                measure_results.append(exec_res[targlist[obi]].calculate_obs(rpos))

        return sum(measure_results)

    def _measure_obs(
        self, qc: QuantumCircuit, measure_base: Optional[List] = None
    ) -> ExecResult:
        """Single run for measurement task.

        Args:
            qc (QuantumCircuit): Quantum circuit that need to be executed on backend.
            measure_base (list[str, list[int]]): measure base and its positions.
        """
        if not isinstance(self._task, Task):
            raise ValueError("_task not initiated in Estimator")

        if measure_base is None:
            res = self._task.send(qc)
            res.measure_base = ""

        else:
            for base, pos in zip(measure_base[0], measure_base[1]):
                if base == "X":
                    qc.ry(pos, -np.pi / 2)
                elif base == "Y":
                    qc.rx(pos, np.pi / 2)

            res = self._task.send(qc)
            res.measure_base = measure_base

        return res

    def _run_simulation(self, observables: Hamiltonian):
        """Run using quafu simulator"""
        # sim_state = simulate(self._circ).get_statevector()
        # expectation = np.matmul(
        #     np.matmul(sim_state.conj().T, observables.get_matrix()), sim_state
        # ).real
        # return expectation
        return execute_circuit(self._circ, observables)

    def run(self, observables: Hamiltonian, params: List[float]):
        """Calculate estimation for given observables

        Args:
            observables: observables to be estimated.
            paras_list: list of parameters of self.circ.

        Returns:
            Expectation value
        """
        if params is not None:
            self._circ._update_params(params)

        if self._backend == "sim":
            return self._run_simulation(observables)
        else:
            return self._run_real_machine(observables)
