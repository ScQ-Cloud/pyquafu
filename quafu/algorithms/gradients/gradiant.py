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
"""Get gradient of parameterized circuit."""
import numpy as np
from quafu.circuits.quantum_circuit import QuantumCircuit
from quafu.elements import ControlledGate, Parameter, ParameterExpression, QuantumGate
from quafu.elements.matrices import XMatrix, YMatrix, ZMatrix
from quafu.exceptions import CircuitError
from quafu.simulators.simulator import SVSimulator


def assemble_grads(para_grads, gate_grads):
    grads = []
    for var in para_grads:
        grad_p = para_grads[var]
        fullgrad = 0.0
        for pos_g in grad_p:
            pos, gp = pos_g
            gg = gate_grads[pos[0]][pos[1]]
            fullgrad += gg * gp
        grads.append(fullgrad)

    return grads


def grad_para_shift(qc: QuantumCircuit, hamiltonian, backend=SVSimulator(), psi_in = np.array([], dtype=complex)):
    """
    Parameter shift gradients. Each gate must have one parameter
    """
    para_grads = qc._calc_parameter_grads()
    gate_grads = [[] for _ in qc.gates]

    for i, op in enumerate(qc.gates):
        if len(op.paras) > 0:
            if isinstance(op.paras[0], (Parameter, ParameterExpression)):
                if op.name not in ["RX", "RY", "RZ"]:
                    raise CircuitError(
                        "It seems the circuit can not apply parameter-shift rule to calculate gradient."
                        " You may need compile the circuit first"
                    )
                op.paras[0] = op.paras[0] + np.pi / 2
                res1 = sum(backend.run(qc, hamiltonian=hamiltonian, psi=psi_in)["pauli_expects"])
                op.paras[0] = op.paras[0] - np.pi
                res2 = sum(backend.run(qc, hamiltonian=hamiltonian, psi=psi_in)["pauli_expects"])
                op.paras[0]._undo(2)
                gate_grads[i].append((res1 - res2) / 2.0)

    return assemble_grads(para_grads, gate_grads)


def grad_finit_diff(qc, hamiltonian, backend=SVSimulator(), psi_in = np.array([], dtype=complex)):
    variables = qc.variables
    grads = []
    for v in variables:
        v.value += 1e-10
        res1 = sum(backend.run(qc, hamiltonian=hamiltonian, psi=psi_in)["pauli_expects"])
        v.value -= 2 * 1e-10
        res2 = sum(backend.run(qc, hamiltonian=hamiltonian, psi=psi_in)["pauli_expects"])
        v.value += 1e-10
        grads.append((res1 - res2) / (2 * 1e-10))

    return grads


def grad_gate(op):
    """
    TODO:support more gates
    """
    if isinstance(op, ControlledGate):
        if op._targ_name == "RX":
            circ = QuantumCircuit(max(op.pos) + 1)
            deriv_mat = -0.5j * XMatrix @ op._get_targ_matrix()
            circ << QuantumGate("dRX", op.targs, [], deriv_mat)
            cdim = 1 << (len(op.ctrls))
            proj_mat = np.zeros((cdim, cdim))
            proj_mat[cdim - 1, cdim - 1] = 1.0
            circ << QuantumGate("projCtrl", op.ctrls, [], proj_mat)
            return circ.wrap()

        if op._targ_name == "RY":
            circ = QuantumCircuit(max(op.pos) + 1)
            deriv_mat = -0.5j * YMatrix @ op._get_targ_matrix()
            circ << QuantumGate("dRY", op.targs, [], deriv_mat)
            cdim = 1 << (len(op.ctrls))
            proj_mat = np.zeros((cdim, cdim))
            proj_mat[cdim - 1, cdim - 1] = 1.0
            circ << QuantumGate("projCtrl", op.ctrls, [], proj_mat)
            return circ.wrap()

        if op._targ_name == "RZ":
            circ = QuantumCircuit(max(op.pos) + 1)
            deriv_mat = -0.5j * ZMatrix @ op._get_targ_matrix()
            circ << QuantumGate("dRZ", op.targs, [], deriv_mat)
            cdim = 1 << (len(op.ctrls))
            proj_mat = np.zeros((cdim, cdim))
            proj_mat[cdim - 1, cdim - 1] = 1.0
            circ << QuantumGate("projCtrl", op.ctrls, [], proj_mat)
            return circ.wrap()
        raise NotImplementedError

    if op.name == "RX":
        deriv_mat = -0.5j * XMatrix @ op.matrix
        return QuantumGate("dRX", op.pos, [], deriv_mat)
    if op.name == "RY":
        deriv_mat = -0.5j * YMatrix @ op.matrix
        return QuantumGate("dRY", op.pos, [], deriv_mat)
    if op.name == "RZ":
        deriv_mat = -0.5j * ZMatrix @ op.matrix
        return QuantumGate("dRZ", op.pos, [], deriv_mat)
    raise NotImplementedError


def grad_adjoint(qc, hamiltonian, psi_in=np.array([], dtype=complex)):
    """
    Reverse mode gradient: arXiv:2009.02823
    """
    para_grads = qc._calc_parameter_grads()
    backend = SVSimulator()
    lam = backend.run(qc, psi=psi_in)["statevector"]
    phi = np.copy(lam)
    lam = backend._apply_hamil(hamiltonian, lam)
    begin = 0
    end = len(qc.gates)
    gate_grads = [[] for _ in range(end)]
    for i, op in enumerate(qc.gates):
        if len(op.paras) > 0 and (isinstance(op.paras[0], (Parameter, ParameterExpression))):
            begin = i
            break

    for i in range(begin, end)[::-1]:
        op = qc.gates[i]
        phi = backend._apply_op(op.dagger(), phi)
        if len(op.paras) > 0 and (isinstance(op.paras[0], (Parameter, ParameterExpression))):
            mu = np.copy(phi)
            mu = backend._apply_op(grad_gate(op), mu)
            gate_grads[i].append(np.real(2.0 * np.inner(lam.conj(), mu)))
        lam = backend._apply_op(op.dagger(), lam)
    return assemble_grads(para_grads, gate_grads)
