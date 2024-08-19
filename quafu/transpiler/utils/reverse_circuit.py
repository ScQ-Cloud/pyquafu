from quafu import QuantumCircuit


def reverse_circuit(circuit: QuantumCircuit):
    """
    This function aims to provide a reversed circuit for a given circuit.

    Args:
        circuit: input circuit that need to be reversed
    Return:
        the reversed circuit aimed to do bidirectional routing
    """
    rev_circuit = QuantumCircuit(circuit.num)

    if circuit.measures:
        for qubit, cbit in circuit.measures.items():
            rev_circuit.measure([qubit], [cbit])

    for gate in circuit.gates[::-1]:
        rev_circuit.add_gate(gate)
    return rev_circuit

