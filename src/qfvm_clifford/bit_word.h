#ifndef TABLEAU_WORD_H_
#define TABLEAU_WORD_H_

#include "utils.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#ifdef USE_SIMD
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

#include <iostream>
#include <sstream>

// A bit_word is a bag of bits, which can be operated by individual CPU
// instructions.
// This template is necessary due to the varying interfaces between native types
// such as uint64_t and intrinsics like __m256i across different architectures
// and operating systems. In certain contexts, operators can be used on __m256i
// values (e.g. a ^= b), while in others this is not possible. The bitword
// template implementations establish a standard set of methods that are
// essential for Clifford's operation, allowing the same code to be compiled
// using either 256-bit or 64-bit registers depending on what is appropriate.
template <size_t word_size> struct bit_word;

/* =================================================== */
/* ================ bit_word operation =============== */

template <size_t word_size>
inline bool operator==(const bit_word<word_size>& left,
                       const bit_word<word_size>& right) {
  return left.to_u64_array() == right.to_u64_array();
}

template <size_t word_size>
inline bool operator!=(const bit_word<word_size>& left,
                       const bit_word<word_size>& right) {
  return !(left == right);
}

template <size_t word_size>
inline bool operator<(const bit_word<word_size>& left,
                      const bit_word<word_size>& right) {
  auto array1 = left.to_u64_array();
  auto array2 = right.to_u64_array();

  for (size_t i = 0; i < array1.size(); ++i) {
    if (array1[i] != array2[i]) {
      return array1[i] < array2[i];
    }
  }
  return false;
}

template <size_t word_size>
inline bool operator==(const bit_word<word_size>& left, int right) {
  return left == bit_word<word_size>(right);
}

template <size_t word_size>
inline bool operator!=(const bit_word<word_size>& left, int right) {
  return left != right;
}

template <size_t word_size>
inline bool operator==(const bit_word<word_size>& left, uint64_t right) {
  return left == bit_word<word_size>(right);
}

template <size_t word_size>
inline bool operator!=(const bit_word<word_size>& left, uint64_t right) {
  return left != right;
}

template <size_t word_size>
inline bool operator==(const bit_word<word_size>& left, int64_t right) {
  return left == bit_word<word_size>(right);
}

template <size_t word_size>
inline bool operator!=(const bit_word<word_size>& left, int64_t right) {
  return left != right;
}

// output 1 for bit 1, . for bit 0
template <size_t word_size>
std::ostream& operator<<(std::ostream& os, const bit_word<word_size>& word) {
  os << "bit_word<" << word_size << ">{";
  auto array1 = word.to_u64_array();
  for (size_t i = 0; i < array1.size(); ++i) {
    for (size_t j = 0; j < 64; ++j) {
      if ((i | j) && (j & 7) == 0) {
        os << ' ';
      }
      // ".1" is a char array
      os << ".1"[(array1[i] >> j) & 1];
    }
  }
  os << "}";
  return os;
}

template <size_t word_size>
inline bit_word<word_size> operator<<(const bit_word<word_size>& word,
                                      int offset) {
  return word.shift(offset);
}

template <size_t word_size>
inline bit_word<word_size> operator>>(const bit_word<word_size>& word,
                                      int offset) {
  return word.shift(-offset);
}

template <size_t word_size>
inline bit_word<word_size> operator<<=(bit_word<word_size>& word, int offset) {
  return word = word.shift(offset);
}

template <size_t word_size>
inline bit_word<word_size> operator>>=(bit_word<word_size>& word, int offset) {
  return word = word.shift(-offset);
}

template <size_t word_size>
inline bit_word<word_size> operator<<(const bit_word<word_size>& word,
                                      uint64_t offset) {
  return word.shift((int)offset);
}

template <size_t word_size>
inline bit_word<word_size> operator>>(const bit_word<word_size>& word,
                                      uint64_t offset) {
  return word.shift(-(int)offset);
}

template <size_t word_size>
inline bit_word<word_size> operator<<=(bit_word<word_size>& word,
                                       uint64_t offset) {
  return word = word.shift((int)offset);
}

template <size_t word_size>
inline bit_word<word_size> operator>>=(bit_word<word_size>& word,
                                       uint64_t offset) {
  return word = word.shift(-(int)offset);
}

template <size_t word_size>
inline bit_word<word_size> operator&(const bit_word<word_size>& word,
                                     uint64_t mask) {
  return word & bit_word<word_size>(mask);
}

template <size_t word_size>
inline bit_word<word_size> operator|(const bit_word<word_size>& word,
                                     uint64_t mask) {
  return word | bit_word<word_size>(mask);
}

template <size_t word_size>
inline bit_word<word_size> operator^(const bit_word<word_size>& word,
                                     uint64_t mask) {
  return word ^ bit_word<word_size>(mask);
}

template <size_t word_size>
inline bit_word<word_size> operator&(const bit_word<word_size>& word,
                                     int64_t mask) {
  return word & bit_word<word_size>(mask);
}

template <size_t word_size>
inline bit_word<word_size> operator|(const bit_word<word_size>& word,
                                     int64_t mask) {
  return word | bit_word<word_size>(mask);
}

template <size_t word_size>
inline bit_word<word_size> operator^(const bit_word<word_size>& word,
                                     int64_t mask) {
  return word ^ bit_word<word_size>(mask);
}

template <size_t word_size>
inline bit_word<word_size> operator&(const bit_word<word_size>& word,
                                     int mask) {
  return word & bit_word<word_size>(mask);
}

template <size_t word_size>
inline bit_word<word_size> operator|(const bit_word<word_size>& word,
                                     int mask) {
  return word | bit_word<word_size>(mask);
}

template <size_t word_size>
inline bit_word<word_size> operator^(const bit_word<word_size>& word,
                                     int mask) {
  return word ^ bit_word<word_size>(mask);
}

/* =================================================== */
/* ================ 64 bit version =================== */
template <> struct bit_word<64> {
  constexpr static size_t WORD_SIZE = 64;
  constexpr static size_t BIT_POW = 6;

  union {
    uint8_t u8[8];
    uint64_t u64[1];
  };

  inline constexpr bit_word<64>() : u64{} {}
  inline constexpr bit_word<64>(uint64_t v) : u64{v} {}
  inline constexpr bit_word<64>(int64_t v) : u64{(uint64_t)v} {}
  inline constexpr bit_word<64>(int v) : u64{(uint64_t)v} {}

  inline operator bool() const { return bool(u64[0]); }
  inline operator int64_t() const { return int64_t(u64[0]); }
  inline operator int() const { return int64_t(*this); }
  inline operator uint64_t() const { return u64[0]; }

  inline bit_word<64>& operator^=(const bit_word<64>& other) {
    u64[0] ^= other.u64[0];
    return *this;
  }

  inline bit_word<64>& operator&=(const bit_word<64>& other) {
    u64[0] &= other.u64[0];
    return *this;
  }

  inline bit_word<64>& operator|=(const bit_word<64>& other) {
    u64[0] |= other.u64[0];
    return *this;
  }

  inline bit_word<64> operator^(const bit_word<64>& other) const {
    return bit_word<64>(u64[0] ^ other.u64[0]);
  }

  inline bit_word<64> operator&(const bit_word<64>& other) const {
    return bit_word<64>(u64[0] & other.u64[0]);
  }

  inline bit_word<64> operator|(const bit_word<64>& other) const {
    return bit_word<64>(u64[0] | other.u64[0]);
  }

  inline bit_word<64> andnot(const bit_word<64>& other) const {
    return bit_word<64>(~u64[0] & other.u64[0]);
  }

  // convert bit word to string
  operator std::string() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
  }

  std::array<uint64_t, 1> to_u64_array() const {
    return std::array<uint64_t, 1>{u64[0]};
  }

  inline bit_word<64> shift(int offset) const {
    auto array1 = u64[0];
    if (64 <= offset || -64 >= offset) {
      return 0;
    } else if (offset > 0) {
      return bit_word<64>(array1 << offset);
    } else {
      return bit_word<64>(array1 >> -offset);
    }
  }

  inline uint16_t count() const { return count_uint64_bits(u64[0]); }

  static void* aligned_malloc(size_t bits) { return malloc(bits); }

  static void aligned_free(void* ptr) { free(ptr); }

  template <uint64_t mask, uint64_t shift>
  static void inplace_transpose_64_step(uint64_t* data, size_t stride) {
    for (size_t k = 0; k < 64; k++) {
      if (k & shift)
        continue;
      uint64_t& x = data[stride * k];
      uint64_t& y = data[stride * (k + shift)];
      uint64_t a = x & mask;
      uint64_t b = x & ~mask;
      uint64_t c = y & mask;
      uint64_t d = y & ~mask;
      x = a | (c << shift);
      y = (b >> shift) | d;
    }
  }

  static void inplace_transpose_square(bit_word<64>* block_start,
                                       size_t stride) {
    inplace_transpose_64_step<0x5555555555555555ull, 1>((uint64_t*)block_start,
                                                        stride);
    inplace_transpose_64_step<0x3333333333333333ull, 2>((uint64_t*)block_start,
                                                        stride);
    inplace_transpose_64_step<0x0F0F0F0F0F0F0F0Full, 4>((uint64_t*)block_start,
                                                        stride);
    inplace_transpose_64_step<0x00FF00FF00FF00FFull, 8>((uint64_t*)block_start,
                                                        stride);
    inplace_transpose_64_step<0x0000FFFF0000FFFFull, 16>((uint64_t*)block_start,
                                                         stride);
    inplace_transpose_64_step<0x00000000FFFFFFFFull, 32>((uint64_t*)block_start,
                                                         stride);
  }
};

#ifdef USE_SIMD
/* =================================================== */
/* ================ 256 bit version ================== */
template <> struct bit_word<256> {
  constexpr static size_t WORD_SIZE = 256;
  constexpr static size_t BIT_POW = 8;

  union {
    uint8_t u8[32];
    uint64_t u64[4];
    __m256i m256;
  };

  inline constexpr bit_word<256>() : m256(__m256i{}) {}
  inline constexpr bit_word<256>(__m256i v) : m256(v) {}
  inline bit_word<256>(uint64_t v) : m256{_mm256_set_epi64x(0, 0, 0, v)} {}
  inline bit_word<256>(int64_t v)
      : m256{_mm256_set_epi64x(-(v < 0), -(v < 0), -(v < 0), v)} {}
  inline bit_word<256>(int v)
      : m256{_mm256_set_epi64x(-(v < 0), -(v < 0), -(v < 0), v)} {}

  inline operator bool() const {
    return bool(u64[0] | u64[1] | u64[2] | u64[3]);
  }
  inline operator int64_t() const {
    auto words = to_u64_array();
    // x86 is little endian default
    int64_t result = int64_t(words[0]);
    uint64_t expected = result < 0 ? uint64_t(-1) : uint64_t(0);
    if (words[1] != expected || words[2] != expected || words[3] != expected) {
      throw std::runtime_error("int64_t overflow");
    }
    return result;
  }
  inline operator int() const { return int64_t(*this); }
  inline operator uint64_t() const {
    if (u64[1] || u64[2] || u64[3]) {
      throw std::runtime_error("uint64_t overflow");
    }
    return u64[0];
  }

  inline bit_word<256>& operator^=(const bit_word<256>& other) {
    m256 = _mm256_xor_si256(m256, other.m256);
    return *this;
  }

  inline bit_word<256>& operator&=(const bit_word<256>& other) {
    m256 = _mm256_and_si256(m256, other.m256);
    return *this;
  }

  inline bit_word<256>& operator|=(const bit_word<256>& other) {
    m256 = _mm256_or_si256(m256, other.m256);
    return *this;
  }

  inline bit_word<256> operator^(const bit_word<256>& other) const {
    return bit_word<256>(_mm256_xor_si256(m256, other.m256));
  }

  inline bit_word<256> operator&(const bit_word<256>& other) const {
    return bit_word<256>(_mm256_and_si256(m256, other.m256));
  }

  inline bit_word<256> operator|(const bit_word<256>& other) const {
    return bit_word<256>(_mm256_or_si256(m256, other.m256));
  }

  inline bit_word<256> andnot(const bit_word<256>& other) const {
    return bit_word<256>(_mm256_andnot_si256(m256, other.m256));
  }

  // convert bit word to string
  operator std::string() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
  }

  std::array<uint64_t, 4> to_u64_array() const {
    return std::array<uint64_t, 4>{u64[0], u64[1], u64[2], u64[3]};
  }

  inline bit_word<256> shift(int offset) const {
    auto array = to_u64_array();
    while (offset <= -64) {
      array[0] = array[1];
      array[1] = array[2];
      array[2] = array[3];
      array[3] = 0;
      offset += 64;
    }

    while (offset >= 64) {
      array[3] = array[2];
      array[2] = array[1];
      array[1] = array[0];
      array[0] = 0;
      offset -= 64;
    }

    __m256i low2high;
    __m256i high2low;
    if (offset < 0) {
      low2high = _mm256_set_epi64x(0, array[3], array[2], array[1]);
      high2low = _mm256_set_epi64x(array[3], array[2], array[1], array[0]);
      offset += 64;
    } else {
      low2high = _mm256_set_epi64x(array[3], array[2], array[1], array[0]);
      high2low = _mm256_set_epi64x(array[2], array[1], array[0], 0);
    }

    uint64_t mask = (uint64_t{1} << offset) - 1;
    low2high = _mm256_slli_epi64(low2high, offset);
    high2low = _mm256_srli_epi64(high2low, 64 - offset);
    // for offset < 0, only w[1] in lower 64 bits is used
    low2high = _mm256_and_si256(low2high, _mm256_set1_epi64x(~mask));
    // for offset > 0, only w[0] in upper 64 bits is used
    high2low = _mm256_and_si256(high2low, _mm256_set1_epi64x(mask));
    return _mm256_or_si256(low2high, high2low);
  }

  inline uint16_t count() const {
    return count_uint64_bits(u64[0]) + count_uint64_bits(u64[1]) +
           count_uint64_bits(u64[2]) + count_uint64_bits(u64[3]);
  }

  static void* aligned_malloc(size_t bits) {
    return _mm_malloc(bits, sizeof(__m256i));
  }

  static void aligned_free(void* ptr) { _mm_free(ptr); }

  template <uint64_t shift>
  static void inplace_transpose_256_step(__m256i mask, __m256i* data,
                                         size_t stride) {
    for (std::size_t k = 0; k < 256; k++) {
      if (k & shift)
        continue;

      __m256i& x = data[stride * k];
      __m256i& y = data[stride * (k + shift)];
      __m256i a = _mm256_and_si256(x, mask);
      __m256i b = _mm256_andnot_si256(mask, x);
      __m256i c = _mm256_and_si256(y, mask);
      __m256i d = _mm256_andnot_si256(mask, y);

      x = _mm256_or_si256(a, _mm256_slli_epi64(c, shift));
      y = _mm256_or_si256(_mm256_srli_epi64(b, shift), d);
    }
  }

  static void inplace_transpose_64_and_128_step(bit_word<256>* data,
                                                size_t stride) {
    uint64_t* u64_ptr = (uint64_t*)data;
    stride <<= 2;
    for (std::size_t k = 0; k < 64; k++) {
      std::swap(u64_ptr[stride * (k + 64 * 0) + 1],
                u64_ptr[stride * (k + 64 * 1) + 0]);
      std::swap(u64_ptr[stride * (k + 64 * 0) + 2],
                u64_ptr[stride * (k + 64 * 2) + 0]);
      std::swap(u64_ptr[stride * (k + 64 * 0) + 3],
                u64_ptr[stride * (k + 64 * 3) + 0]);
      std::swap(u64_ptr[stride * (k + 64 * 1) + 2],
                u64_ptr[stride * (k + 64 * 2) + 1]);
      std::swap(u64_ptr[stride * (k + 64 * 1) + 3],
                u64_ptr[stride * (k + 64 * 3) + 1]);
      std::swap(u64_ptr[stride * (k + 64 * 2) + 3],
                u64_ptr[stride * (k + 64 * 3) + 2]);
    }
  }

  static void inplace_transpose_square(bit_word<256>* data, size_t stride) {
    inplace_transpose_256_step<1>(_mm256_set1_epi8(0x55), (__m256i*)data,
                                  stride);
    inplace_transpose_256_step<2>(_mm256_set1_epi8(0x33), (__m256i*)data,
                                  stride);
    inplace_transpose_256_step<4>(_mm256_set1_epi8(0x0F), (__m256i*)data,
                                  stride);
    inplace_transpose_256_step<8>(_mm256_set1_epi16(0x00FF), (__m256i*)data,
                                  stride);
    inplace_transpose_256_step<16>(_mm256_set1_epi32(0x0000FFFF),
                                   (__m256i*)data, stride);
    inplace_transpose_256_step<32>(_mm256_set1_epi64x(0x00000000FFFFFFFFull),
                                   (__m256i*)data, stride);
  }
};
#endif

#endif
