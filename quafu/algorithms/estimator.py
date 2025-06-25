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
"""Pre-build wrapper to calculate expectation value."""

import copy
import time
from typing import List, Optional

import numpy as np
from quafu.algorithms.ansatz import QuantumNeuralNetwork
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
        **task_options,
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

        # Caching expectation calculation results
        self._exp_cache = {}

    def _run_real_machine(
        self, observables: Hamiltonian, cache_key: Optional[str] = None
    ):
        """
        Execute the circuit with observable expectation measurement task.
        Args:
            qc (QuantumCircuit): Quantum circuit that need to be executed on backend.
            obslist (list[str, list[int]]): List of pauli string and its position.
            cache_key: if set, check if cache hit and use cached measurement results.

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

        for obs in obslist:
            for p in obs[1]:
                if p not in measures:
                    raise CircuitError(
                        f"Qubit {p} in observer {obs[0]} is not measured."
                    )

        measure_basis, targlist = merge_measure(obslist)
        print("Job start, need measured in ", measure_basis)

        exec_res = []
        if cache_key is not None and cache_key in self._exp_cache:
            # try to retrieve exe results from cache
            exec_res = self._exp_cache[cache_key]
        else:
            # send tasks to cloud platform
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

            if cache_key is not None:
                # put into cache
                self._exp_cache[cache_key] = exec_res

        measure_results = []
        for obi, obs in enumerate(obslist):
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
        return execute_circuit(self._circ, observables)

    def clear_cache(self):
        """clean expectation cache"""
        self._exp_cache.clear()

    def run(
        self,
        observables: Hamiltonian,
        params: List[float],
        cache_key: Optional[str] = None,
    ):
        """Calculate estimation for given observables

        Args:
            observables: observables to be estimated.
            params: list of parameters of self.circ.
            cache_key: if this value is set, we will first look into the _exp_cache to see
                if previous measurement results can be reused. Note that it is the user's duty
                to guarantee correctness.
        Returns:
            Expectation value
        """
        if params is not None:
            if isinstance(
                self._circ, QuantumNeuralNetwork
            ):  # currently circ with this attr is QuantumNeuralNetwork
                # For QuantumNeuralNetwork after v0.4.4, circ structure is determined at runtime
                # thus only update tunable parameters, i.e., weights
                if not self._circ.is_legacy_if():
                    # In this case, we are handling QuantumNeuralNetwork after v0.4.4
                    # First we extract the input parameters for embedding
                    dim_weights = self._circ.num_tunable_parameters
                    dim_embed_in = len(params) - dim_weights
                    embed_in = params[:dim_embed_in]
                    # Then we re-construct the circuit
                    self._circ.reconstruct(embed_in)
                    num_params = len(self._circ.variables)
                    # Finally we update tunable parameters
                    new_params = np.zeros((num_params,))
                    new_params[-dim_weights:] = params[-dim_weights:]
                    self._circ._update_params(new_params, tunable_only=True)
                else:
                    self._circ._update_params(params)
            else:
                self._circ._update_params(params)

        if self._backend == "sim":
            return self._run_simulation(observables)
        return self._run_real_machine(observables, cache_key=cache_key)
