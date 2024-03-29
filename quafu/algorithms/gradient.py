from ..circuits.quantum_circuit import QuantumCircuit
from ..simulators.simulator import SVSimulator
from ..elements import Parameter, ParameterExpression
import numpy as np
from ..exceptions import CircuitError
from ..elements.matrices import XMatrix, YMatrix, ZMatrix
from ..elements import QuantumGate, ControlledGate

def assemble_grads(para_grads, gate_grads):
    grads = []
    for var in para_grads:
        grad_p = para_grads[var]
        fullgrad = 0.
        for pos_g in grad_p:
            pos, gp = pos_g 
            gg = gate_grads[pos[0]][pos[1]]
            fullgrad += gg * gp
        grads.append(fullgrad)
    
    return grads

def grad_para_shift(qc:QuantumCircuit, hamiltonian, backend=SVSimulator()):
    """
    Parameter shift gradients. Each gate must have one parameter
    """
    para_grads = qc._calc_parameter_grads()
    gate_grads= [[] for _ in qc.gates]

    for i, op in enumerate(qc.gates):
        if len(op.paras) > 0:
            if isinstance(op.paras[0], Parameter) or isinstance(op.paras[0],ParameterExpression):
                if op.name not in ["RX", "RY", "RZ"]:
                    raise CircuitError("It seems the circuit can not apply parameter-shift rule to calculate gradient.You may need compile the circuit first")
                op.paras[0] = op.paras[0] + np.pi/2
                res1 = sum(backend.run(qc, hamiltonian=hamiltonian)["pauli_expects"])
                op.paras[0] = op.paras[0] - np.pi
                res2 = sum(backend.run(qc, hamiltonian=hamiltonian)["pauli_expects"])
                op.paras[0]._undo(2)
                gate_grads[i].append((res1 - res2) / 2.)
        
    return assemble_grads(para_grads, gate_grads)

def grad_finit_diff(qc, hamiltonian, backend=SVSimulator()):
    variables = qc.variables
    grads = []
    for v in variables:
        v.value += 1e-10
        res1 = sum(backend.run(qc, hamiltonian=hamiltonian)["pauli_expects"])
        v.value -= 2 * 1e-10
        res2 = sum(backend.run(qc, hamiltonian=hamiltonian)["pauli_expects"])
        v.value += 1e-10
        grads.append((res1 - res2) / (2 * 1e-10))
    
    return grads


def grad_gate(op):
    """
    TODO:support more gates
    """
    if isinstance(op, ControlledGate):
        if op._targ_name == "RX":
            circ = QuantumCircuit(max(op.pos)+1)
            deriv_mat = -0.5j * XMatrix @ op._get_targ_matrix()
            circ << QuantumGate("dRX", op.targs, [], deriv_mat)
            cdim = 1 << (len(op.ctrls))
            proj_mat = np.zeros((cdim, cdim))
            proj_mat[cdim-1, cdim-1] = 1.
            circ << QuantumGate("projCtrl", op.ctrls, [], proj_mat)
            return circ.wrap()
        
        elif op._targ_name == "RY":
            circ = QuantumCircuit(max(op.pos)+1)
            deriv_mat = -0.5j * YMatrix @ op._get_targ_matrix()
            circ << QuantumGate("dRY", op.targs, [], deriv_mat)
            cdim = 1 << (len(op.ctrls))
            proj_mat = np.zeros((cdim, cdim))
            proj_mat[cdim-1, cdim-1] = 1.
            circ << QuantumGate("projCtrl", op.ctrls, [], proj_mat)
            return circ.wrap()
        
        elif op._targ_name == "RZ":
            circ = QuantumCircuit(max(op.pos)+1)
            deriv_mat = -0.5j * ZMatrix @ op._get_targ_matrix()
            circ << QuantumGate("dRZ", op.targs, [], deriv_mat)
            cdim = 1 << (len(op.ctrls))
            proj_mat = np.zeros((cdim, cdim))
            proj_mat[cdim-1, cdim-1] = 1.
            circ << QuantumGate("projCtrl", op.ctrls, [], proj_mat)
            return circ.wrap()
        else:
            raise NotImplementedError
        
    else:
        if op.name == "RX":
            deriv_mat = -0.5j * XMatrix @ op.matrix
            return QuantumGate("dRX", op.pos, [], deriv_mat)
        elif op.name == "RY":
            deriv_mat = -0.5j * YMatrix @ op.matrix
            return QuantumGate("dRY", op.pos, [], deriv_mat)
        elif op.name == "RZ":
            deriv_mat = -0.5j * ZMatrix @ op.matrix
            return QuantumGate("dRZ", op.pos, [], deriv_mat)
        else:
            raise NotImplementedError
    
def grad_adjoint(qc, hamiltonian, psi_in=np.array([], dtype=complex)):
    """
    Reverse mode gradient: arXiv:2009.02823
    """
    para_grads = qc._calc_parameter_grads()
    backend = SVSimulator()
    lam = backend.run(qc, psi = psi_in)["statevector"]
    phi = np.copy(lam)
    lam = backend._apply_hamil(hamiltonian, lam)
    begin = 0
    end = len(qc.gates)
    gate_grads= [[] for _ in range(end)]
    for i, op in enumerate(qc.gates):
        if len(op.paras) > 0 and (isinstance(op.paras[0], Parameter) or isinstance(op.paras[0],ParameterExpression)):
            begin = i
            break
    
    for i in range(begin, end)[::-1]:
        op = qc.gates[i]
        phi = backend._apply_op(op.dagger(), phi)
        if len(op.paras) > 0 and (isinstance(op.paras[0], Parameter) or isinstance(op.paras[0],ParameterExpression)):
                mu = np.copy(phi)
                mu = backend._apply_op(grad_gate(op), mu)
                gate_grads[i].append(np.real(2. * np.inner(lam.conj(), mu)))
        lam = backend._apply_op(op.dagger(), lam)
    return assemble_grads(para_grads, gate_grads)