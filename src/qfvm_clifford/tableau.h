#ifndef TABLEAU_H_
#define TABLEAU_H_

#include "gate_macro.h"
#include "packed_bit_word.h"
#include "pauli.h"
#include "pauli_slice.h"
#include "table.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <math.h>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

// tableau trans is the transpose of the tableau
template <size_t word_size> struct tableau_trans;

// tableau is the main class of the clifford simulator
template <size_t word_size> struct tableau;

// quantum gate type
// SINGLE_QUBIT_GATE: single qubit gate
// TWO_QUBIT_GATE: two qubit gate
// COLLAPSING_GATE: measurement gate
// ERROR_QUBIT_GATE: error gate, which has some probability to apply the gate
enum gate_type {
  SINGLE_QUBIT_GATE,
  TWO_QUBIT_GATE,
  COLLAPSING_GATE,
  ERROR_QUBIT_GATE
};

// quantum gate function return type
// std::nullopt: The return value of non-measurement quantum gate
// (COLLASPING_GATE) bool: The return value of measurement quantum gate
using result = std::optional<bool>;

// quantum gate function type
// The type is used to represent the all the quantum gate function type, which
// is for store gate map. The first type is for single qubit gate and error
// qubit gate, the second type is for two qubit gate, the third type is for
// collasping gate
template <size_t word_size>
using func_type = std::variant<
    std::function<result(tableau<word_size>& t, const size_t qubit)>,
    std::function<result(tableau<word_size>& t, const size_t qubit1,
                         const size_t qubit2)>,
    std::function<result(tableau<word_size>& t, std::mt19937_64& rng,
                         const size_t qubit)>>;

// the implementation of the quantum gate function
#define FUNCTION_REGISTRATION
#include "gate_list.h"
#undef FUNCTION_REGISTRATION

// the gate map, which is used to store all the quantum gate function, the index
// is the gate name
template <size_t word_size>
std::unordered_map<std::string, std::pair<gate_type, func_type<word_size>>>
    gate_map = {
#define GATE_MAP_REGISTRATION
#include "gate_list.h"
#undef GATE_MAP_REGISTRATION
};

// Inner struct of the tableau, which is used to store the the distabilizer and
// stabilizer of the tableau.
//
// Reference: https://arxiv.org/pdf/quant-ph/0406196.pdf
template <size_t word_size> struct _tableau {
  size_t num_qubits;

  // The stabilizer tableau is represented by two bit tables, one for X and one
  // for Z.
  table<word_size> xs_t, zs_t;

  // The signs of the tableau
  packed_bit_word<word_size> signs;

  // constructor, n is the number of qubits
  _tableau(size_t n) : num_qubits(n), xs_t(n, n), zs_t(n, n), signs(n) {}

  // index operator
  pauli_string_slice<word_size> operator[](size_t qubit) {
    // if padding, the number of real words is not the same with the number of
    // padded words
    size_t num_words = (num_qubits + word_size - 1) / word_size;
    return pauli_string_slice<word_size>(num_qubits, signs[qubit],
                                         xs_t[qubit].slice(0, num_words),
                                         zs_t[qubit].slice(0, num_words));
  }

  const pauli_string_slice<word_size> operator[](size_t qubit) const {
    size_t num_words = (num_qubits + word_size - 1) / word_size;
    return pauli_string_slice<word_size>(num_qubits, signs[qubit],
                                         xs_t[qubit].slice(0, num_words),
                                         zs_t[qubit].slice(0, num_words));
  }
};

// A Clifford operation is a unitary quantum operation that conjugates Pauli
// products into Pauli products. C is Clifford if, for all pauli products P, it
// is the case that C^*PC is also a Pauli product. In fact, a Clifford operation
// can be uniquely identified (up to global phase) by how it conjugates Pauli
// products.
// A stabilizer tableau is a representation of a Clifford operation
// that simply directly stores how the Clifford operation conjugates each
// generator of the Pauli group.
template <size_t word_size> struct tableau {

  size_t num_qubits;

  // n distabilizer generators which are Pauli operators that together with the
  // stabilizer generators generate the full Pauli group
  _tableau<word_size> distabilizer;
  _tableau<word_size> stabilizer;

  // constructor
  explicit tableau(size_t num_qubits)
      : num_qubits(num_qubits), distabilizer(num_qubits),
        stabilizer(num_qubits) {
    // Initialize identity elements along the diagonal. The state is |0...0>
    for (size_t q = 0; q < num_qubits; q++) {
      distabilizer.xs_t[q][q] = true;
      stabilizer.zs_t[q][q] = true;
    }
  }

  // convert tableau to string
  operator std::string() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
  }

// quantum gate function registration
#define STRUCT_FUNCTION_REGISTRATION
#include "gate_list.h"
#undef STRUCT_FUNCTION_REGISTRATION

  std::string str() const { return std::string(*this); }

  // return identity tableau, that means the state is |0...0>
  static tableau<word_size> identity(size_t num_qubits) {
    return tableau<word_size>(num_qubits);
  }

  void reset() { *this = tableau<word_size>(num_qubits); }
  void reset(std::mt19937_64& rng, size_t qubit) { r_gate(rng, qubit); }
  // void reset_x(std::mt19937_64& rng, size_t qubit) { rx_gate(rng, qubit); }
  // void reset_y(std::mt19937_64& rng, size_t qubit) { ry_gate(rng, qubit); }

  // expand current tableau to new_num_qubits
  // args:
  //   new_num_qubits: the new number of qubits
  //   resize_pad_factor: the resize pad factor, which means leave more space
  //   for future storage of more quantum bits.
  void expand(size_t new_num_qubits, double resize_pad_factor) {

    assert(new_num_qubits >= num_qubits);
    assert(resize_pad_factor >= 1);

    if (new_num_qubits <= distabilizer.xs_t.num_bit_words_major * word_size) {
      size_t old_num_qubits = num_qubits;
      num_qubits = new_num_qubits;
      distabilizer.num_qubits = new_num_qubits;
      stabilizer.num_qubits = new_num_qubits;

      // Initialize identity elements along the diagonal.
      for (size_t k = old_num_qubits; k < new_num_qubits; k++) {
        distabilizer.xs_t[k][k] = true;
        stabilizer.zs_t[k][k] = true;
      }

      return;
    }

    size_t old_num_bit_words = distabilizer.xs_t.num_bit_words_major;
    size_t old_num_qubits = num_qubits;
    tableau<word_size> old_state = std::move(*this);
    *this = tableau<word_size>((size_t)(new_num_qubits * resize_pad_factor));
    this->num_qubits = new_num_qubits;
    this->distabilizer.num_qubits = new_num_qubits;
    this->stabilizer.num_qubits = new_num_qubits;

    // Copy stored state back into new larger space.
    auto partial_copy = [=](packed_bit_word_slice<word_size> dst,
                            packed_bit_word_slice<word_size> src) {
      dst.slice(0, old_num_bit_words) = src;
    };
    partial_copy(distabilizer.signs, old_state.distabilizer.signs);
    partial_copy(stabilizer.signs, old_state.stabilizer.signs);
    for (size_t k = 0; k < old_num_qubits; k++) {
      partial_copy(distabilizer[k].xs, old_state.distabilizer[k].xs);
      partial_copy(distabilizer[k].zs, old_state.distabilizer[k].zs);
      partial_copy(stabilizer[k].xs, old_state.stabilizer[k].xs);
      partial_copy(stabilizer[k].zs, old_state.stabilizer[k].zs);
    }
  }

  // transpose each table of the tableau
  void inplace_transpose() {
    stabilizer.xs_t.inplace_transpose();
    stabilizer.zs_t.inplace_transpose();
    distabilizer.xs_t.inplace_transpose();
    distabilizer.zs_t.inplace_transpose();
  }

  // Clifford state measurements only have three probabilities: (p0, p1) = (0.5,
  // 0.5), (1, 0), or (0, 1) The random case happens if there is a row
  // anti-commuting with Z[qubit]
  bool is_deterministic_z(size_t target_qubit) const {
    return !stabilizer[target_qubit].xs.is_not_all_zero();
  }

  bool is_deterministic_x(size_t target_qubit) const {
    return !distabilizer[target_qubit].xs.is_not_all_zero();
  }

  bool is_deterministic_y(size_t target_qubit) const {
    return distabilizer[target_qubit].xs == stabilizer[target_qubit].xs;
  }

  pauli_string<word_size> eval_y_obs(size_t qubit) const {
    pauli_string<word_size> result = distabilizer[qubit];
    uint8_t log_i = pauli_string_slice<word_size>(result).inplace_right_mul(
        stabilizer[qubit]);
    log_i++;
    assert((log_i & 1) == 0);
    if (log_i & 2) {
      result.sign ^= true;
    }
    return result;
  }

  // collapse the qubit along z axis
  // args:
  //   t_trans: the transpose of the tableau
  //   target_qubit: the target qubit
  //   rng: the random number generator
  size_t collapse_qubit_along_z(tableau_trans<word_size>& t_trans,
                                size_t target_qubit, std::mt19937_64& rng) {

    size_t pivot = 0;

    // search for any generator that anti-commutes with the measurement
    // observable
    while (pivot < num_qubits &&
           !t_trans.t.stabilizer.xs_t[pivot][target_qubit])
      pivot++;

    // Such an p does not exist. In this case the outcome is determinate, so
    // measuring the state will not change it; the only task is to determine
    // whether 0 or 1 is observed.
    if (pivot == num_qubits)
      return SIZE_MAX;

    // perform partial gaussian elimination over the stabilizer generators that
    // anti-commute with the measurement. do this by introducing
    // no-effect-because-control-is-zero CNOT at the beginning of time.
    for (size_t k = pivot + 1; k < num_qubits; k++)
      if (t_trans.t.stabilizer.xs_t[k][target_qubit])
        t_trans.cnot_gate(pivot, k);

    // swap the non-isolated anti-commuting stablizer generator for one that
    // commutes with the measurement
    if (t_trans.t.stabilizer.zs_t[pivot][target_qubit]) {
      t_trans.h_yz_gate(pivot);
    } else {
      t_trans.h_gate(pivot);
    }

    // assign measure result
    bool result_if_measured = rng() & 1;
    if (stabilizer.signs[target_qubit] != result_if_measured) {
      t_trans.x_gate(pivot);
    };

    return pivot;
  }

  // random valid stablizer tableau
  // reference: https://arxiv.org/abs/2003.09412
  static tableau<word_size>
  random_valid_stabilizer_tableau(size_t num_qubits, std::mt19937_64& rng) {
    auto raw = table<word_size>::random_valid_stabilizer_table(num_qubits, rng);
    tableau<word_size> result(num_qubits);
    for (size_t row = 0; row < num_qubits; row++) {
      for (size_t col = 0; col < num_qubits; col++) {
        result.distabilizer[row].xs[col] = raw[row][col];
        result.distabilizer[row].zs[col] = raw[row][col + num_qubits];
        result.stabilizer[row].xs[col] = raw[row + num_qubits][col];
        result.stabilizer[row].zs[col] =
            raw[row + num_qubits][col + num_qubits];
      }
    }

    result.distabilizer.signs.randomize(num_qubits, rng);
    result.stabilizer.signs.randomize(num_qubits, rng);
    return result;
  }

  // check whether the tableau satisfy the invariants, the tableau need to
  // preserve commutativity
  // everything must commute, except for X_k anticommuting with Z_k for each k.
  bool satisfy_invariants() const {
    for (size_t q1 = 0; q1 < num_qubits; q1++) {
      auto x1 = distabilizer[q1];
      auto z1 = stabilizer[q1];

      if (x1.commutes(z1))
        return false;

      for (size_t q2 = q1 + 1; q2 < num_qubits; q2++) {
        auto x2 = distabilizer[q2];
        auto z2 = stabilizer[q2];

        if (!x1.commutes(x2) || !x1.commutes(z2) || !z1.commutes(x2) ||
            !z1.commutes(z2))
          return false;
      }
    }

    return true;
  }
};

template <size_t word_size>
std::ostream& operator<<(std::ostream& out, const tableau<word_size>& t) {
  out << "+-";
  for (size_t k = 0; k < t.num_qubits; k++) {
    out << "xz-";
  }
  out << "+\n|";
  for (size_t k = 0; k < t.num_qubits; k++) {
    out << ' ' << "+-"[t.distabilizer[k].sign] << "+-"[t.stabilizer[k].sign];
  }
  for (size_t q = 0; q < t.num_qubits; q++) {
    out << " |\n|";
    for (size_t k = 0; k < t.num_qubits; k++) {
      out << ' '
          << "IXZY"[t.distabilizer[k].xs[q] | t.distabilizer[k].zs[q] << 1]
          << "IXZY"[t.stabilizer[k].xs[q] | t.stabilizer[k].zs[q] << 1];
    }
  }
  out << " |";
  return out;
}

// reference to the tableau, transpose the tableau at the construction, after
// some computation, transpose back at the deconstruction
template <size_t word_size> struct tableau_trans {
  // referece to the tableau
  tableau<word_size>& t;

  // constructor
  explicit tableau_trans(tableau<word_size>& t_in) : t(t_in) {
    t.inplace_transpose();
  };
  tableau_trans() = delete;

  // copt and move constructor
  tableau_trans(const tableau_trans<word_size>& t) = delete;
  tableau_trans(tableau_trans<word_size>&& t) = delete;

  // deconstructor
  ~tableau_trans() { t.inplace_transpose(); }

  // Iterates over the Paulis in a row of the tableau.
  //
  // args
  //   q: The row to iterate over.
  //   body: A function taking X, Z, and SIGN words. The X and Z words are
  //   chunks of xz-encoded Paulis from the row. The SIGN word is the
  //   corresponding chunk of sign bits from the sign row.
  template <typename FUNC>
  inline void for_each_trans_obs(const size_t q, FUNC body) {
    for (size_t k = 0; k < 2; k++) {
      _tableau<word_size>& h = k == 0 ? t.distabilizer : t.stabilizer;
      pauli_string_slice<word_size> p = h[q];
      p.xs.for_each_word(p.zs, h.signs, body);
    }
  }

  template <typename FUNC>
  inline void for_each_trans_obs(const size_t q1, const size_t q2, FUNC body) {
    for (size_t k = 0; k < 2; k++) {
      _tableau<word_size>& h = k == 0 ? t.distabilizer : t.stabilizer;
      pauli_string_slice<word_size> p1 = h[q1];
      pauli_string_slice<word_size> p2 = h[q2];
      p1.xs.for_each_word(p1.zs, p2.xs, p2.zs, h.signs, body);
    }
  }

  tableau_trans<word_size>& x_gate(const size_t qubit) {
    for_each_trans_obs(qubit, [](auto& x, auto& z, auto& s) { s ^= z; });
    return *this;
  }

  tableau_trans<word_size>& h_gate(const size_t qubit) {
    for_each_trans_obs(qubit, [](auto& x, auto& z, auto& s) {
      std::swap(x, z);
      s ^= x & z;
    });
    return *this;
  }

  tableau_trans<word_size>& h_yz_gate(const size_t qubit) {
    for_each_trans_obs(qubit, [](auto& x, auto& z, auto& s) {
      s ^= z.andnot(x);
      x ^= z;
    });
    return *this;
  }

  tableau_trans<word_size>& cnot_gate(const size_t control_qubit,
                                      const size_t target_qubit) {
    for_each_trans_obs(control_qubit, target_qubit,
                       [](auto& cx, auto& cz, auto& tx, auto& tz, auto& s) {
                         s ^= (cz ^ tx).andnot(cx & tz);
                         cz ^= tz;
                         tx ^= cx;
                       });
    return *this;
  }
};
#endif
