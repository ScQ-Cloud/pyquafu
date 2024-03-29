# Clifford Simulator

Usage of Clifford simulator in ``pyquafu`` is very similar with the default simulator, but applies only to circuits with a limited gate set.

## Supported Gates

### x

The Pauli X gate. The bit flip gate.

### y

The Pauli Y gate.

### z

The Pauli Z gate. The phase flip gate

### h

The Hadamard gate. Swap the X and Z axes. Alternate name: `h_xz`

### h_yz

The variant of the Hadamard gate that swaps the Y and Z axes (instead of X and Z)

### s

Principal square root of Z gate. Phases the amplitue of |1> by i. Alternate name: `sqrt_z`

### s_dag

Adjoint of the principal square root of z gate. Phases the amplitube of |1> by -i. Alternate name: `sqrt_z_dag`

### cnot

The Z-controlled X gate. Applies an X gate to the target if the control is in the |1> state. Equivalently: negates the amplitude of the |1>|-> state. The first qubit is called the control, and the second qubit is the target. Alternate name: `cx`, `zcx`

### swap

Swap two qubits.

### measure

Z-basis measurement. Projects each target qubit into |0> or |1> and reports its value (false=|0>, true=|1>).

### reset

Z-basis reset. Forces each target qubit into the |0> state by silently measuring it in the Z basis and applying an X gate if it ended up in the |1> state.

## Simple Example

```python
from quafu import QuantumCircuit, simulate

qc = QuantumCircuit(4, 4)
qc.x(0)
qc.x(1)
qc.measure([1, 2], [1, 0])
qc.measure([3, 0], [2, 3])

result = simulate(qc=qc, shots=10, simulator="clifford")
counts = result.count

# {"0101": 10}
print(counts)
```
