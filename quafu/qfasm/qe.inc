
// 2-parameter 1-pulse single qubit gate
gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
// controlled-NOT
gate cx c,t { CX c,t; }
// Clifford gate: Hadamard
gate h a { u2(0,pi) a; }
gate u3(theta,phi,lambda) q 
{ 
  U(theta,phi,lambda) q; 
}
gate uni_test a {u2(0,pi) a;}