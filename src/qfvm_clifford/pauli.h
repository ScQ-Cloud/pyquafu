#ifndef PAULI_H_
#define PAULI_H_

#include "bit.h"
#include "packed_bit_word.h"
#include "pauli_slice.h"
#include "table.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <ostream>
#include <random>
#include <string>

// a pauli string is a product of pauli operators (I, X, Y, Z) on n qubits
template <size_t word_size> struct pauli_string {
  // the length of the pauli string
  size_t num_qubits;

  // whether the pauli string is a negative, true if negative and false if
  // positive
  bool sign;

  // the paulis in the pauli string, paulis are encoded by xz (I=00, X=10, Y=11,
  // Z=01)
  packed_bit_word<word_size> xs, zs;

  explicit pauli_string(size_t n) : num_qubits(n), sign(false), xs(n), zs(n) {}

  explicit pauli_string(std::string& str)
      : num_qubits(0), sign(false), xs(0), zs(0) {
    *this = std::move(from_cstr(str.c_str()));
  }

  // copy constructor
  pauli_string(const pauli_string<word_size>& other)
      : num_qubits(other.num_qubits), sign(other.sign), xs(other.xs),
        zs(other.zs) {}

  pauli_string(const pauli_string_slice<word_size>& other)
      : num_qubits(other.num_qubits), sign(other.sign), xs(other.xs),
        zs(other.zs) {}

  // move constructor
  pauli_string(pauli_string<word_size>&& other) noexcept
      : num_qubits(other.num_qubits), sign(other.sign), xs(std::move(other.xs)),
        zs(std::move(other.zs)) {}

  // copy assignment
  pauli_string<word_size>& operator=(const pauli_string<word_size>& other) {
    this->num_qubits = other.num_qubits, this->sign = other.sign;
    this->xs = other.xs, this->zs = other.zs;

    return *this;
  }

  pauli_string<word_size>&
  operator=(const pauli_string_slice<word_size>& other) {
    this->num_qubits = other.num_qubits, this->sign = other.sign;
    this->xs = other.xs, this->zs = other.zs;

    return *this;
  }

  // move assignment
  pauli_string<word_size>& operator=(pauli_string<word_size>&& other) {
    this->~pauli_string();
    new (this) pauli_string<word_size>(std::move(other));

    return *this;
  }

  // equality operator
  bool operator==(const pauli_string<word_size>& other) const {
    return num_qubits == other.num_qubits && bool(sign) == bool(other.sign) &&
           xs == other.xs && zs == other.zs;
  }

  bool operator==(const pauli_string_slice<word_size>& other) const {
    return num_qubits == other.num_qubits && bool(sign) == bool(other.sign) &&
           xs == other.xs && zs == other.zs;
  }

  bool operator!=(const pauli_string<word_size>& other) const {
    return !(*this == other);
  }

  bool operator!=(const pauli_string_slice<word_size>& other) const {
    return !(*this == other);
  }

  // convert to pauli string slice
  operator const pauli_string_slice<word_size>() const {
    return pauli_string_slice<word_size>(num_qubits, bit((void*)&sign, 0), xs,
                                         zs);
  }

  operator pauli_string_slice<word_size>() {
    return pauli_string_slice<word_size>(num_qubits, bit(&sign, 0), xs, zs);
  }

  // convert pauli string to string
  std::string str() const { return std::string(*this); }

  operator std::string() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
  }

  // generate a random pauli string
  static pauli_string<word_size> random(size_t num_qubits,
                                        std::mt19937_64& rng) {
    auto result = pauli_string<word_size>(num_qubits);
    result.xs.randomize(num_qubits, rng);
    result.zs.randomize(num_qubits, rng);
    result.sign ^= rng() & 1;
    return result;
  }

  // parse char one by one
  static pauli_string<word_size>
  parse_cstr(size_t num_qubits, bool sign,
             const std::function<char(size_t)>& func) {
    pauli_string<word_size> ret(num_qubits);
    ret.sign = sign;

    for (size_t i = 0; i < num_qubits; i++) {
      bool x, z;
      switch (func(i)) {
      case 'X':
        x = true, z = false;
        break;
      case 'Y':
        x = true, z = true;
        break;
      case 'Z':
        x = false, z = true;
        break;
      case 'I':
        x = false, z = false;
        break;
      default:
        throw std::invalid_argument("can not parse character: " +
                                    std::to_string(func(i)));
      }

      ret.xs.u64[i / 64] ^= (uint64_t)x << (i & 63);
      ret.zs.u64[i / 64] ^= (uint64_t)z << (i & 63);
    }

    return ret;
  }

  // convert c style string to pauli string
  static pauli_string<word_size> from_cstr(const char* cstr) {
    // default is positive
    auto sign = cstr[0] == '-';
    if ('-' == cstr[0] || '+' == cstr[0])
      cstr++;

    return parse_cstr(strlen(cstr), sign, [&](size_t i) { return cstr[i]; });
  }
};

template <size_t word_size>
std::ostream& operator<<(std::ostream& os, const pauli_string<word_size>& str) {
  os << (str.sign ? '-' : '+');
  for (size_t i = 0; i < str.num_qubits; i++) {
    os << "IXZY"[str.xs[i] | (str.zs[i] << 1)];
  }
  return os;
}

#endif
