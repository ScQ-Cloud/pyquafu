OPENQASM 2.0;
gate tan(theta) c,t { cx c,t; u1(tan(theta)) c;}
qreg q[3];
qreg r[3];
h q;
cx q, r;
cx q, r;
tan(1) r, q;
creg c[3];
creg d[3];
//barrier q;
u2(tan(1), 1) q;
measure q->c;
measure r->d;