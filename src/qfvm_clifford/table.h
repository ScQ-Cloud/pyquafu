#ifndef TABLE_H_
#define TABLE_H_

#include "bit.h"
#include "bit_word.h"
#include "packed_bit_word.h"
#include "packed_bit_word_slice.h"
#include "utils.h"
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <random>
#include <sstream>
#include <string>

// row-major table, padded and aligned to make table more efficient
// major represents the row (not contiguous in memory), minor represents the
// column (contiguous in memory), the smallest table is word_size * word_size
template <size_t word_size> struct table {
  size_t num_bit_words_major;
  size_t num_bit_words_minor;

  packed_bit_word<word_size> data;

  table(size_t min_bits_major, size_t min_bits_minor)
      : num_bit_words_major(bits_to_word_padded<word_size>(min_bits_major)),
        num_bit_words_minor(bits_to_word_padded<word_size>(min_bits_minor)),
        data(bits_to_bits_padded<word_size>(min_bits_major) *
             bits_to_bits_padded<word_size>(min_bits_minor)) {}

  // index operator, major index should be the number of rows
  inline packed_bit_word_slice<word_size> operator[](size_t major_index) {
    return data.slice(major_index * num_bit_words_minor, num_bit_words_minor);
  }

  // index operator, major index should be the number of rows
  inline const packed_bit_word_slice<word_size>
  operator[](size_t major_index) const {
    return data.slice(major_index * num_bit_words_minor, num_bit_words_minor);
  }

  // equality operator
  bool operator==(const table<word_size>& other) const {
    return num_bit_words_major == other.num_bit_words_major &&
           num_bit_words_minor == other.num_bit_words_minor &&
           data == other.data;
  }

  bool operator!=(const table<word_size>& other) const {
    return !(*this == other);
  }

  // convert stablizer tableau to string
  std::string str() const { return std::string(*this); }

  // for better printing
  std::string str(size_t n) const {
    std::stringstream ss;
    for (size_t i = 0; i < n; i++) {
      if (i)
        ss << "\n";
      for (size_t j = 0; j < n; j++)
        ss << "_1"[(*this)[i][j]];
    }

    return ss.str();
  }

  operator std::string() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
  }

  // major_index is the index of bit_word in the row, minor_index is the index
  // of bit_word in the column, major_index_sub is the index of bit in the
  // bit_word
  inline size_t get_index_bit_word(const size_t major_index,
                                   const size_t minor_index,
                                   const size_t major_index_sub) const {
    auto index =
        (major_index << bit_word<word_size>::BIT_POW) + major_index_sub;
    return index * num_bit_words_minor + minor_index;
  }

  // transpose the table
  table<word_size> transpose() const {
    table<word_size> result(num_bit_words_minor * word_size,
                            num_bit_words_major * word_size);

    for (size_t major_word = 0; major_word < num_bit_words_major;
         major_word++) {
      for (size_t minor_word = 0; minor_word < num_bit_words_minor;
           minor_word++) {
        for (size_t major_word_sub = 0; major_word_sub < word_size;
             major_word_sub++) {
          size_t src_index =
              get_index_bit_word(major_word, minor_word, major_word_sub);
          size_t dst_index =
              result.get_index_bit_word(minor_word, major_word, major_word_sub);
          result.data.bw[dst_index] = data.bw[src_index];
        }
      }
    }

    // transpose the bit word block, the shape of the block is (word_size,
    // word_size)
    for (size_t major_word = 0; major_word < result.num_bit_words_major;
         major_word++) {
      for (size_t minor_word = 0; minor_word < result.num_bit_words_minor;
           minor_word++) {
        size_t block_start =
            result.get_index_bit_word(major_word, minor_word, 0);
        bit_word<word_size>::inplace_transpose_square(
            result.data.bw + block_start, result.num_bit_words_minor);
      }
    }

    return result;
  }

  // inplace transpose, only for square matrix
  table<word_size>& inplace_transpose() {

    // transpose the bit word block, the shape of the block is (word_size,
    // word_size)
    for (size_t major_word = 0; major_word < num_bit_words_major;
         major_word++) {
      for (size_t minor_word = 0; minor_word < num_bit_words_minor;
           minor_word++) {
        size_t block_start = get_index_bit_word(major_word, minor_word, 0);
        bit_word<word_size>::inplace_transpose_square(data.bw + block_start,
                                                      num_bit_words_minor);
      }
    }

    // transpose the table
    for (size_t major_word = 0; major_word < num_bit_words_major; major_word++)
      for (size_t minor_word = major_word + 1; minor_word < num_bit_words_minor;
           minor_word++)
        for (size_t major_word_sub = 0; major_word_sub < word_size;
             major_word_sub++)
          std::swap(data.bw[get_index_bit_word(major_word, minor_word,
                                               major_word_sub)],
                    data.bw[get_index_bit_word(minor_word, major_word,
                                               major_word_sub)]);

    return *this;
  }

  // square matrix multiplication (assuming row indexing)
  table<word_size> square_matrix_mul(const table<word_size>& right,
                                     size_t n) const {
    auto tmp = right.transpose();

    table<word_size> result(n, n);
    for (std::size_t i = 0; i < n; i++) {
      for (std::size_t j = 0; j < n; j++) {
        bit_word<word_size> accumulater{};
        (*this)[i].for_each_word(
            tmp[j], [&](auto& a, auto& b) { accumulater ^= a & b; });
        result[i][j] = accumulater.count() & 1;
      }
    }

    return result;
  }

  // sqaure matrix inverse for lower triangular matrix
  table<word_size> inverse_for_lower_triangular_matrix(size_t n) const {
    table<word_size> result = table<word_size>::identity(n);
    packed_bit_word<word_size> tmp(num_bit_words_minor * word_size);

    for (size_t i = 0; i < n; i++) {
      tmp = (*this)[i];
      // pivot
      for (size_t j = 0; j < i; j++) {
        if (tmp[j]) {
          tmp ^= (*this)[j];
          result[i] ^= result[j];
        }
      }
    }

    return result;
  }

  // concatenate four tables
  static table<word_size>
  concatenate_four(size_t n, const table<word_size>& upper_left,
                   const table<word_size>& upper_right,
                   const table<word_size>& lower_left,
                   const table<word_size>& lower_right) {
    table<word_size> result(n << 1, n << 1);
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < n; j++) {
        result[i][j] = upper_left[i][j];
        result[i][j + n] = upper_right[i][j];
        result[i + n][j] = lower_left[i][j];
        result[i + n][j + n] = lower_right[i][j];
      }
    }

    return result;
  }

  // generate identity table
  static table<word_size> identity(size_t n) {
    table<word_size> result(n, n);
    for (size_t i = 0; i < n; i++)
      result[i][i] = true;

    return result;
  }

  // generate random table
  static table<word_size> random(size_t random_bits_major,
                                 size_t random_bits_minor,
                                 std::mt19937_64& rng) {
    table<word_size> result(random_bits_major, random_bits_minor);
    for (size_t major = 0; major < random_bits_major; major++)
      result[major].randomize(random_bits_minor, rng);

    return result;
  }

  // Sample from the quantum Mallows distribution, generate a bit string h and a
  // permutation S
  // TODO: fix -Wstringop-overflow in test when n is 1
  static inline std::pair<std::vector<bool>, std::vector<size_t>>
  sample_quantum_mallows(size_t n, std::mt19937_64& rng) {
    auto r_dis = std::uniform_real_distribution<double>(0, 1);
    std::vector<bool> h;
    std::vector<size_t> S;
    std::vector<size_t> A;

    for (size_t i = 0; i < n; i++)
      A.push_back(i);

    for (size_t i = 0; i < n; i++) {
      auto m = A.size();
      auto r = r_dis(rng);
      auto eps = pow(4, -int(m));
      auto k = size_t(-ceil(log2(r + (1 - r) * eps)));
      h.push_back(k < m);
      if (k >= m)
        k = 2 * m - k - 1;
      S.push_back(A[k]);
      A.erase(A.begin() + k);
    }

    return {h, S};
  }

  // Samples a random valid stabilizer tableau.
  // reference: Generation of random Clifford operators in
  // https://arxiv.org/pdf/2003.09412.pdf
  static table<word_size> random_valid_stabilizer_table(size_t n,
                                                        std::mt19937_64& rng) {
    auto h_S = sample_quantum_mallows(n, rng);

    const auto& h = h_S.first;
    const auto& S = h_S.second;

    table<word_size> symmetric(n, n);
    for (size_t i = 0; i < n; i++) {
      symmetric[i].randomize(i + 1, rng);
      for (size_t j = 0; j < i; j++)
        symmetric[j][i] = symmetric[i][j];
    }

    table<word_size> symmetric_m(n, n);
    for (size_t i = 0; i < n; i++) {
      symmetric_m[i].randomize(i + 1, rng);
      symmetric_m[i][i] &= h[i];
      for (size_t j = 0; j < i; j++) {
        bool b = h[i] && h[j];
        b |= h[i] > h[j] && S[i] < S[j];
        b |= h[i] < h[j] && S[i] > S[j];
        symmetric_m[i][j] &= b;
        symmetric_m[j][i] = symmetric_m[i][j];
      }
    }

    auto lower = table<word_size>::identity(n);
    for (size_t i = 0; i < n; i++)
      lower[i].randomize(i, rng);

    auto lower_m = table<word_size>::identity(n);
    for (size_t i = 0; i < n; i++) {
      lower_m[i].randomize(i, rng);
      for (size_t j = 0; j < i; j++) {
        bool b = h[i] < h[j];
        b |= h[i] && h[j] && S[i] > S[j];
        b |= !h[i] && !h[j] && S[i] < S[j];
        lower_m[i][j] &= b;
      }
    }

    // a normalized probability distribution, P_n(h, S) is the fraction of
    // n-qubit Clifford operators U such that the canonical form of U defined in
    // Theorem 1 contains a layer of h gates labeled by h and a qubit
    // permutation S.
    auto prod = symmetric.square_matrix_mul(lower, n);
    auto prod_m = symmetric_m.square_matrix_mul(lower_m, n);

    auto inv = lower.inverse_for_lower_triangular_matrix(n);
    auto inv_m = lower_m.inverse_for_lower_triangular_matrix(n);

    inv.inplace_transpose();
    inv_m.inplace_transpose();

    // the first n columns represent Pauli operators Fx_iF^{-1} (ignoring the
    // phase) and the last n columns represent Fz_iF^{−1} . Stabilizer tableau
    // of the Hadamard stage and qubit permutation layers in the canonical form
    auto fused = table<word_size>::concatenate_four(
        n, lower, table<word_size>(n, n), prod, inv);
    auto fused_m = table<word_size>::concatenate_four(
        n, lower_m, table<word_size>(n, n), prod_m, inv_m);

    table<word_size> u(2 * n, 2 * n);
    for (size_t i = 0; i < n; i++) {
      u[i] = fused[S[i]];
      u[i + n] = fused[S[i] + n];
    }

    // hadamards
    for (size_t i = 0; i < n; i++)
      if (h[i])
        u[i].swap(u[i + n]);

    return fused_m.square_matrix_mul(u, 2 * n);
  }
};

template <size_t word_size>
std::ostream& operator<<(std::ostream& os, const table<word_size>& table) {
  for (size_t i = 0; i < table.num_bit_words_major; i++) {
    if (i)
      os << "\n";
    os << table[i];
  }

  return os;
}

#endif
