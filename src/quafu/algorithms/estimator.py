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

from typing import Optional
from quafu import QuantumCircuit
from quafu.simulators.simulator import simulate
from quafu.tasks.tasks import Task

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
        if task is not None:
            self._task = task
            self._task.config(backend=self._backend)
            self._task.config(**task_options)

    def _run_real_machine(self, observables):
        """TODO"""
        if not isinstance(self._task, Task):
            raise ValueError("task not set")
        res, obsexp = self._task.submit(self._circ, observables)
        return res, obsexp

    def _run_simulation(self, observables):
        """TODO"""
        sim_res = simulate(self._circ)
        # TODO
        # sim_res.calculate_obs()

    def run(self, observables, paras_list):
        """Calculate estimation for given observables

        Args:
            observables: observables to be estimated.
            paras_list: list of parameters of self.circ.
        """
        self._circ.update_params(paras_list)

        if self._backend == "sim":
            self._run_simulation(observables)
        else:
            self._run_real_machine(observables)
