OPENQASM 2.0;
include "qe.inc";
gate test c,t { cx x,t; }
qreg q[3];
qreg r[3];
h q;
cx q, r;
uni_test q;
test r, q;
creg c[3];
creg d[3];
barrier q;
u2(tan(1), 1) q;
measure q->c;
measure r->d;
//u3(1,2,3) r;
//if(c == 7) u3(0,0,0) q;