OPENQASM 2.0;
include "qe.inc";
gate test c,t { cx c,t; }
qreg q[3];
qreg r[3];
h q;
cx q, r;
uni_test q;
test q, r;
creg c[3];
creg d[3];
barrier q;
measure q->c;
measure r->d;
u2(tan(1), 0) q;