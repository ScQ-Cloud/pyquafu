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
from typing import List, Optional
from ..circuits.quantum_circuit import QuantumCircuit
from ..tasks.tasks import Task
from .hamiltonian import Hamiltonian
from ..simulators import simulate


def execute_circuit(circ: QuantumCircuit, observables: Hamiltonian):
    """Execute circuit on quafu simulator"""
    sim_res = simulate(circ, hamiltonian= observables)
    expectations = sim_res["pauli_expects"]
    return sum(expectations)


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
        self._backend = backend
        self._task = None
        if backend != "sim":
            if task is not None:
                self._task = task
            else:
                self._task = Task()
            self._task.config(backend=self._backend)
            self._task.config(**task_options)

    def _run_real_machine(self, observables: Hamiltonian):
        """Submit to quafu service"""
        if not isinstance(self._task, Task):
            raise ValueError("task not set")
        # TODO(zhaoyilun): replace old `submit` API in the future,
        #   investigate the best implementation for calculating
        #   expectation on real devices.
        obs = observables.to_legacy_quafu_pauli_list()
        _, obsexp = self._task.submit(self._circ, obs)
        return sum(obsexp)

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
            self._circ.update_params(params)

        if self._backend == "sim":
            return self._run_simulation(observables)
        else:
            return self._run_real_machine(observables)
