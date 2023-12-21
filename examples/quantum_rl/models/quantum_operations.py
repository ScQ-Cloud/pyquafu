import cirq
import numpy as np
import sympy

# Organize components defined below to quantum operations
OPS = {
    "variationalPQC": lambda qubits, position, params: generate_vpqc(
        qubits, position, params
    ),
    "dataencodingPQC": lambda qubits, position, count, params, state: generate_dpqc(
        qubits, position, count, params, state
    ),
    "entanglement": lambda qubits: generate_entangle(qubits),
}


def one_qubit_rotation(qubit, symbols):
    """
    Return Cirq gates that apply a rotation of the bloch sphere about the X,
    Y and Z axis, specified by the values in `symbols`.
    """
    return [
        cirq.rx(symbols[0])(qubit),
        cirq.ry(symbols[1])(qubit),
        cirq.rz(symbols[2])(qubit),
    ]


def entangling_layer(qubits):
    """
    Return a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    Note: for lower depth of compiled circuits, you can only choose adjacent CZ
    """
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    # cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cz_ops


def generate_vpqc(qubits, position, params=None):
    """Prepare a variational circuit on `qubits` at `position`."""
    # Number of qubits
    n_qubits = len(qubits)

    # Sympy symbols or load parameters for variational angles
    if params == None:
        params = sympy.symbols(
            f"theta({3*position*n_qubits}:{3*(position+1)*n_qubits})"
        )
    else:
        params = params[3 * position * n_qubits : 3 * (position + 1) * n_qubits]
    params = np.asarray(params).reshape((n_qubits, 3))

    # Define circuit
    circuit = cirq.Circuit()

    # Variational layer
    circuit += cirq.Circuit(
        one_qubit_rotation(q, params[i]) for i, q in enumerate(qubits)
    )

    return circuit, list(params.flat)


def generate_dpqc(qubits, position, count, params=None, state=None):
    """Prepare a dataencoding circuit on `qubits` at `position`."""
    # Number of qubits
    n_qubits = len(qubits)

    # Sympy symbols or load parameters for encoding angles
    if params == None:
        inputs = sympy.symbols(f"x{position}" + f"_(0:{n_qubits})")
    else:
        inputs = params[count * n_qubits : (count + 1) * n_qubits]
        for i in range(len(state)):
            inputs[i] *= state[i]
    inputs = np.asarray(inputs).reshape((n_qubits))

    # Define circuit
    circuit = cirq.Circuit()

    # Encoding layer
    circuit += cirq.Circuit(cirq.rx(inputs[i])(q) for i, q in enumerate(qubits))

    return circuit, list(inputs.flat)


def generate_entangle(qubits):
    """Prepare a entangle circuit on `qubits`."""
    # Define circuit
    circuit = cirq.Circuit()

    circuit += entangling_layer(qubits)

    return circuit
