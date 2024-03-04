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
"""simulator for quantum circuit"""

from ..elements import CircuitWrapper, QuantumGate, KrausChannel, UnitaryChannel
from ..circuits import QuantumCircuit
from abc  import ABC, abstractmethod
from .qfvm import simulate_circuit, applyop_statevec, expect_statevec, sampling_statevec,simulate_circuit_clifford
import numpy as np
from ..exceptions import QuafuError
from ..results.results import SimuResult
from ..algorithms.hamiltonian import Hamiltonian

class Simulator(ABC):
    @abstractmethod
    def run(self):
        raise NotImplementedError
    
class SVSimulator(Simulator):
    def __init__(self, use_gpu:bool=False, use_custatevec:bool=False):
        self.use_gpu  = use_gpu
        self.use_custatevec = use_custatevec

    def config(self, **kwargs):
        for attr, value in kwargs.items():
            if hasattr(self, attr):
                setattr(self, attr, value)
            else:
                raise ValueError("No such attribute")

    def _apply_op(self, op, psi):
        #TODO: support GPU
        if isinstance(op,CircuitWrapper):
            psi = self.run(op.circuit, psi)["statevector"]
            return psi
        elif isinstance(op, QuantumGate):
            psi = applyop_statevec(op, psi)
            return psi
        elif isinstance(op, KrausChannel):
            temppsi = np.copy(psi)
            norm0 = np.linalg.norm(psi)
            s = 0.
            r = np.random.rand()
            for kop in op.gatelist:
                temppsi = applyop_statevec(kop, temppsi)
                norm1 = np.linalg.norm(temppsi)
                s += norm1 / norm0
                if r < s:
                    psi = temppsi
                    psi = psi / norm1
                    return psi
                else:
                    temppsi = np.copy(psi)
            return psi
        else:
            raise NotImplementedError
    
    def _apply_hamil(self, hamil, psi):
        psi_out = np.zeros(len(psi), dtype=complex) 
        for pauli in hamil.paulis:
            psi1 = np.copy(psi)
            for name, pos in zip(pauli.paulistr, pauli.pos):
                op = QuantumGate.gate_classes[name.lower()](pos)
                psi1 = self._apply_op(op, psi1)
            psi1 = psi1 * pauli.coeff
            psi_out += psi1
    
        return psi_out

    def run(self, qc : QuantumCircuit, psi : np.ndarray= np.array([]), shots:int=0, hamiltonian:Hamiltonian=None):
        res_info = {}
        if qc.noised:
            raise QuafuError("Can not run noisy circuits with statevector simulator, please use the noisy version.")
        
        if self.use_gpu:
            if qc.executable_on_backend == False:
                raise QuafuError("classical operation do not support gpu currently")

            if self.use_custatevec:
                try:
                    from .qfvm import simulate_circuit_custate
                except ImportError:
                    raise QuafuError("pyquafu isn't installed with cuquantum support")
                psi = simulate_circuit_custate(qc, psi)
                count_dict = sampling_statevec(qc.measures, psi, shots)
                res_info["statevector"] = psi
                res_info["counts"] = count_dict
            else:
                try:
                    from .qfvm import simulate_circuit_gpu
                except ImportError:
                    raise QuafuError("you are not using the GPU version of pyquafu")
                psi = simulate_circuit_gpu(qc, psi)
                count_dict = sampling_statevec(qc.measures, psi, shots)
        else:
            count_dict, psi = simulate_circuit(qc, psi, shots)
            res_info["statevector"] = psi
            res_info["counts"] = count_dict

        if hamiltonian:
            paulis = hamiltonian.paulis
            res = expect_statevec(res_info["statevector"], paulis)
            for i in range(len(paulis)):
                res[i] *= paulis[i].coeff
            res_info["pauli_expects"] = res
        else:
            res_info["pauli_expects"] = []
        res_info["qbitnum"] = qc.num
        res_info["measures"] =  qc.measures
        res_info["simulator"] = "statevector"
        return SimuResult(res_info)

class NoiseSVSimulator(Simulator):
    def __init__(self, use_gpu:bool=False, use_custatevec:bool=False):
        self.backend = SVSimulator(use_gpu=use_gpu, use_custatevec=use_custatevec)

    def run_once(self, qc : QuantumCircuit, psi, hamiltonian=None):
        newqc = self.gen_circuit(qc)
        for op in newqc.instructions:
            psi = self.backend._apply_op(op, psi)
        sample = list(sampling_statevec(qc.measures, psi, 1).keys())[0]

        if hamiltonian:
            paulis = hamiltonian.paulis
            res = expect_statevec(psi, paulis)
            for i in range(len(paulis)):
                res[i] *= paulis[i].coeff
            return sample, res
        return sample, None

    def run(self, qc:QuantumCircuit,  psi : np.ndarray= np.array([]), shots:int=0, hamiltonian:Hamiltonian=None):
        counts = {}
        pauli_expects = 0.
        
        for _ in range(shots):
            if not psi:
                tpsi = np.zeros(2**qc.num, dtype=complex)
                tpsi[0] = 1.0
            else:
                tpsi = np.copy(psi)
            sample, pauli_res = self.run_once(qc, tpsi, hamiltonian)
            if sample in counts.keys():
                counts[sample] += 1
            else:
                counts[sample] = 1
            if hamiltonian:
                pauli_expects += np.array(pauli_res)
        pauli_expects /= shots
        if not hamiltonian:
            pauli_expects = []
        else:
            pauli_expects = list(pauli_expects)
        res_info = {"counts":counts, "pauli_expects": pauli_expects}
        res_info["qbitnum"] = qc.num
        res_info["measures"] =  qc.measures 
        res_info["simulator"] = "noisy statevector"
        return SimuResult(res_info)

    @staticmethod
    def gen_circuit(qc):
        """
        sample circuit from noise circuit
        """ 
        num = qc.num
        new_qc = QuantumCircuit(num)
        temp_qc = QuantumCircuit(num)
        if qc._has_wrap:
            qc.unwarp()

        for op in qc.instructions:
            if isinstance(op, QuantumGate):
                temp_qc << op
            elif isinstance(op, UnitaryChannel):
                g = op.gen_gate()
                if g.name != "ID":
                    temp_qc << g
            elif isinstance(op, KrausChannel):
                new_qc << temp_qc.wrap()
                new_qc << op
                temp_qc =  QuantumCircuit(num)
        new_qc << temp_qc.wrap()
        return new_qc
    
class CliffordSimulator(Simulator):
     def run(self, qc : QuantumCircuit, shots:int=0):
        res_info = {}
        count_dict = simulate_circuit_clifford(qc, shots)
        res_info["qbitnum"] = qc.num
        res_info["counts"] = count_dict
        res_info["measures"] = qc.measures
        res_info["simuator"] = "clifford"
        return SimuResult(res_info)
     
