# (C) Copyright 2024 Beijing Academy of Quantum Information Sciences
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

from quafu import QuantumCircuit

n = 4
qc = QuantumCircuit(n)

qasm = """
// Generated from Cirq v0.8.0

OPENQASM 2.0;
include "qelib1.inc";

// Qubits: [0, 1, 2, 3]
qreg q[4];
creg c[4];

x q[0];
x q[1];

// Gate: PhasedISWAP**0.9951774602384953
rz(pi*0.25) q[1];
rz(pi*-0.25) q[2];
cx q[1],q[2];
h q[1];
cx q[2],q[1];
rz(pi*0.4975887301) q[1];
cx q[2],q[1];
rz(pi*-0.4975887301) q[1];
h q[1];
cx q[1],q[2];
rz(pi*-0.25) q[1];
rz(pi*0.25) q[2];

rz(0) q[2];

// Gate: PhasedISWAP**-0.5024296754026449
rz(pi*0.25) q[0];
rz(pi*-0.25) q[1];
cx q[0],q[1];
h q[0];
cx q[1],q[0];
rz(pi*-0.2512148377) q[0];
cx q[1],q[0];
rz(pi*0.2512148377) q[0];
h q[0];
cx q[0],q[1];
rz(pi*-0.25) q[0];
rz(pi*0.25) q[1];

rz(0) q[1];

// Gate: PhasedISWAP**-0.49760685888033646
rz(pi*0.25) q[2];
rz(pi*-0.25) q[3];
cx q[2],q[3];
h q[2];
cx q[3],q[2];
rz(pi*-0.2488034294) q[2];
cx q[3],q[2];
rz(pi*0.2488034294) q[2];
h q[2];
cx q[2],q[3];
rz(pi*-0.25) q[2];
rz(pi*0.25) q[3];

rz(0) q[3];

// Gate: PhasedISWAP**0.004822678143889672
rz(pi*0.25) q[1];
rz(pi*-0.25) q[2];
cx q[1],q[2];
h q[1];
cx q[2],q[1];
rz(pi*0.0024113391) q[1];
cx q[2],q[1];
rz(pi*-0.0024113391) q[1];
h q[1];
cx q[1],q[2];
rz(pi*-0.25) q[1];
rz(pi*0.25) q[2];

rz(0) q[2];

measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
"""

qc.from_openqasm(qasm)
qc.draw_circuit()
qc.plot_circuit(show=True, title="Variational n4")
