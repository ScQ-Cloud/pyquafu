OPENQASM 2.0;
include "qelib1.inc";
qreg a[4];
creg c[4];
h a;
measure a->b;
cnot a;
