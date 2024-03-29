#ifndef PAULI_SLICE_H_
#define PAULI_SLICE_H_

#include "bit.h"
#include "packed_bit_word_slice.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <sstream>

// reference to a slice of a pauli string
template <size_t word_size> struct pauli_string_slice {
  size_t num_qubits;
  bit sign;
  packed_bit_word_slice<word_size> xs, zs;

  // num_qubits should be the same with the number of padded bit_words, that
  // means, (num_qubits + word_size - 1 / word_size) == xs.num_bit_words ==
  // zs.num_bit_words
  pauli_string_slice(size_t num_qubits, bit sign,
                     packed_bit_word_slice<word_size> xs,
                     packed_bit_word_slice<word_size> zs)
      : num_qubits(num_qubits), sign(sign), xs(xs), zs(zs) {}

  // assign operator
  pauli_string_slice<word_size>&
  operator=(const pauli_string_slice<word_size>& other) {
    num_qubits = other.num_qubits;
    sign = other.sign;
    xs = other.xs;
    zs = other.zs;

    return *this;
  }

  // mulitply a commuting pauli string
  pauli_string_slice<word_size>&
  operator*=(const pauli_string_slice<word_size>& other) {
    auto res = inplace_right_mul(other);
    // must be commute
    assert((res & 1) == 0);
    // if the result phase is positive, then compute excluse-or of the signs
    sign ^= res & 2;
    return *this;
  }

  // mulitply a pauli string, ignore anti-commuting terms
  pauli_string_slice<word_size>&
  mul_ignore_anti_commute(const pauli_string_slice<word_size>& other) {
    auto res = inplace_right_mul(other);
    // if the result phase is positive, then compute excluse-or of the signs
    sign ^= res & 2;
    return *this;
  }

  // euqality operator
  bool operator==(const packed_bit_word_slice<word_size>& other) {
    return num_qubits == other.num_qubits && sign == other.sign &&
           xs == other.xs && zs == other.zs;
  }

  bool operator!=(const packed_bit_word_slice<word_size>& other) {
    return !(*this == other);
  }

  // convert pauli string to string
  operator std::string() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
  }

  // swap operator, swap signs, x matrix and z matrix
  void swap(pauli_string_slice<word_size> other) {
    sign.swap(other.sign);
    xs.swap(other.xs);
    zs.swap(other.zs);
  }

  // Intuitively, this functionreturns the exponent to which the imaginary
  // number i is raised when the Pauli matrices represented by x1z1 and x2z2
  // are multiplied together. For example, if x1 = z2 = 0 and z1 = x2 = 1 then
  // Definition 2 shows that x1z1 and x2z2 represent Z and X, respectively.
  // Multiplying Z and X together gives ZX = iY . Since the exponent on i is 1,
  // the result of this function is 1.
  // Returns:
  //  0 if the product is 1
  //  1 if the product is i
  //  2 if the product is -1
  //  3 if the product is -i
  uint8_t
  inplace_right_mul(const pauli_string_slice<word_size>& other) noexcept {
    bit_word<word_size> count1{};
    bit_word<word_size> count2{};

    xs.for_each_word(
        zs, other.xs, other.zs,
        [&count1, &count2](auto& x1, auto& z1, auto& x2, auto& z2) {
          // accumulate anti-commutation (+i or -i) counts
          auto x1z2 = x1 & z2;
          auto anti_commutes = (x2 & z1) ^ x1z2;

          // update left side pauli
          x1 ^= x2;
          z1 ^= z2;

          // accumulate anti-commutation (+i or -i) counts
          count2 ^= (count1 ^ x1 ^ z1 ^ x1z2) & anti_commutes;
          count1 ^= anti_commutes;
        });

    // combine final anti-commutation phase tally (mod 4)
    auto s = count1.count();
    s ^= count2.count() << 1;
    s ^= other.sign << 1;
    return s & 3;
  }

  // determines if the pauli string commutes with the given pauli string
  bool commutes(const pauli_string_slice<word_size>& other) const noexcept {
    if (num_qubits > other.num_qubits)
      return other.commutes(*this);

    bit_word<word_size> count{};
    xs.for_each_word(zs, other.xs, other.zs,
                     [&count](auto& x1, auto& z1, auto& x2, auto& z2) {
                       count ^= (x1 & z2) ^ (x2 & z1);
                     });
    return (count.count() & 1) == 0;
  }
};

template <size_t word_size>
std::ostream& operator<<(std::ostream& os,
                         const pauli_string_slice<word_size> str) {
  os << "+-"[str.sign];
  for (size_t i = 0; i < str.num_qubits; i++) {
    os << "IXZY"[str.xs[i] | (str.zs[i] << 1)];
  }

  return os;
}

#endif
