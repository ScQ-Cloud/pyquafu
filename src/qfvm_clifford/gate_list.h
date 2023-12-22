#include "gate_macro.h"
#include "tableau.h"
#include <type_traits>

SINGLE_QUBIT_GATE(h, {
  t.distabilizer[qubit].swap(t.stabilizer[qubit]);
  return {};
})

SINGLE_QUBIT_GATE(i, { return {}; })

SINGLE_QUBIT_GATE(x, {
  t.stabilizer[qubit].sign ^= true;
  return {};
})

SINGLE_QUBIT_GATE(y, {
  t.distabilizer[qubit].sign ^= true;
  t.stabilizer[qubit].sign ^= true;
  return {};
})

SINGLE_QUBIT_GATE(z, {
  t.distabilizer[qubit].sign ^= true;
  return {};
})

ERROR_QUBIT_GATE(x_error, {
  t.stabilizer[qubit].sign ^= true;
  return {};
})

ERROR_QUBIT_GATE(y_error, {
  t.distabilizer[qubit].sign ^= true;
  t.stabilizer[qubit].sign ^= true;
  return {};
})

ERROR_QUBIT_GATE(z_error, {
  t.distabilizer[qubit].sign ^= true;
  return {};
})

SINGLE_QUBIT_GATE(h_yz, {
  t.stabilizer[qubit].mul_ignore_anti_commute(t.distabilizer[qubit]);
  t.z_gate(qubit);
  return {};
})

SINGLE_QUBIT_GATE(s_dag, {
  t.distabilizer[qubit].mul_ignore_anti_commute(t.stabilizer[qubit]);
  return {};
})

SINGLE_QUBIT_GATE(s, {
  t.s_dag_gate(qubit);
  t.z_gate(qubit);
  return {};
})

// control, target
TWO_QUBIT_GATE(cnot, {
  t.stabilizer[qubit2] *= t.stabilizer[qubit1];
  t.distabilizer[qubit1] *= t.distabilizer[qubit2];
  return {};
})

// control, target
TWO_QUBIT_GATE(cx, {
  t.stabilizer[qubit2] *= t.stabilizer[qubit1];
  t.distabilizer[qubit1] *= t.distabilizer[qubit2];
  return {};
})

TWO_QUBIT_GATE(swap, {
  t.stabilizer[qubit1].swap(t.stabilizer[qubit2]);
  t.distabilizer[qubit1].swap(t.distabilizer[qubit2]);
  return {};
})

// Z-basis measurement. Projects each target qubit into |0> or |1> and reports
// its value (false=|0>, true=|1>).
COLLAPSING_GATE(m, {
  if (!t.is_deterministic_z(qubit)) {

    tableau_trans<word_size> t_trans(t);
    t.collapse_qubit_along_z(t_trans, qubit, rng);
  }

  return t.stabilizer.signs[qubit];
})

// Z-basis measurement. Projects each target qubit into |0> or |1> and reports
// its value (false=|0>, true=|1>).
COLLAPSING_GATE(measure, {
  if (!t.is_deterministic_z(qubit)) {

    tableau_trans<word_size> t_trans(t);
    t.collapse_qubit_along_z(t_trans, qubit, rng);
  }

  return t.stabilizer.signs[qubit];
})

// Z-basis measurement. Projects each target qubit into |0> or |1> and reports
// its value (false=|0>, true=|1>).
COLLAPSING_GATE(mz, {
  if (!t.is_deterministic_z(qubit)) {

    tableau_trans<word_size> t_trans(t);
    t.collapse_qubit_along_z(t_trans, qubit, rng);
  }

  return t.stabilizer.signs[qubit];
})

// Y-basis measurement. Projects each target qubit into |i> or |-i> and reports
// its value (false=|i>, true=|-i>).
COLLAPSING_GATE(my, {
  if (!t.is_deterministic_y(qubit)) {

    t.h_gate(qubit);
    tableau_trans<word_size> t_trans(t);
    t.collapse_qubit_along_z(t_trans, qubit, rng);
    t.h_gate(qubit);
  }

  return t.eval_y_obs(qubit).sign;
})

// X-basis measurement. Projects each target qubit into |+> or |-> and reports
// its value (false=|+>, true=|->).
COLLAPSING_GATE(mx, {
  if (!t.is_deterministic_x(qubit)) {

    t.h_yz_gate(qubit);
    tableau_trans<word_size> t_trans(t);
    t.collapse_qubit_along_z(t_trans, qubit, rng);
    t.h_yz_gate(qubit);
  }

  return t.distabilizer.signs[qubit];
})

// Z-basis reset. Forces each target qubit into the |0> state by silently
// measuring it in the Z basis and applying an X gate if it ended up in the |1>
// state.
COLLAPSING_GATE(r, {
  // Collapse the qubits to be reset.
  if (!t.is_deterministic_z(qubit)) {

    tableau_trans<word_size> t_trans(t);
    t.collapse_qubit_along_z(t_trans, qubit, rng);
  }

  // Force the collapsed qubits into the ground state.
  t.distabilizer.signs[qubit] = false;
  t.stabilizer.signs[qubit] = false;

  return {};
})

// Z-basis reset. Forces each target qubit into the |0> state by silently
// measuring it in the Z basis and applying an X gate if it ended up in the |1>
// state.
COLLAPSING_GATE(reset, {
  // Collapse the qubits to be reset.
  if (!t.is_deterministic_z(qubit)) {

    tableau_trans<word_size> t_trans(t);
    t.collapse_qubit_along_z(t_trans, qubit, rng);
  }

  // Force the collapsed qubits into the ground state.
  t.distabilizer.signs[qubit] = false;
  t.stabilizer.signs[qubit] = false;

  return {};
})

// // Z-basis reset. Forces each target qubit into the |0> state by silently
// // measuring it in the Z basis and applying an X gate if it ended up in the
// |1>
// // state.
// COLLAPSING_GATE(rz, {
//   // Collapse the qubits to be reset.
//   if (!t.is_deterministic_z(qubit)) {

//     tableau_trans<word_size> t_trans(t);
//     t.collapse_qubit_along_z(t_trans, qubit, rng);
//   }

//   // Force the collapsed qubits into the ground state.
//   t.distabilizer.signs[qubit] = false;
//   t.stabilizer.signs[qubit] = false;

//   return {};
// })

// // X-basis reset. Forces each target qubit into the |+> state by silently
// // measuring it in the X basis and applying a Z gate if it ended up in the
// |->
// // state.
// COLLAPSING_GATE(rx, {
//   // Collapse the qubits to be reset.
//   if (!t.is_deterministic_x(qubit)) {

//     t.h_yz_gate(qubit);
//     tableau_trans<word_size> t_trans(t);
//     t.collapse_qubit_along_z(t_trans, qubit, rng);
//     t.h_yz_gate(qubit);
//   }

//   // Force the collapsed qubits into the ground state.
//   t.distabilizer.signs[qubit] = false;
//   t.stabilizer.signs[qubit] = false;

//   return {};
// })

// // Y-basis reset. Forces each target qubit into the |i> state by silently
// // measuring it in the Y basis and applying an X gate if it ended up in the
// |-i>
// // state.
// COLLAPSING_GATE(ry, {
//   // Collapse the qubits to be reset.
//   if (!t.is_deterministic_y(qubit)) {

//     t.h_gate(qubit);
//     tableau_trans<word_size> t_trans(t);
//     t.collapse_qubit_along_z(t_trans, qubit, rng);
//     t.h_gate(qubit);
//   }

//   // Force the collapsed qubits into the ground state.
//   t.distabilizer.signs[qubit] = false;
//   t.stabilizer.signs[qubit] = false;
//   t.stabilizer.signs[qubit] ^= t.eval_y_obs(qubit).sign;

//   return {};
// })

#undef SINGLE_QUBIT_GATE
#undef TWO_QUBIT_GATE
#undef COLLAPSING_GATE
#undef ERROR_QUBIT_GATE
