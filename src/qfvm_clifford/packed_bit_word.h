#ifndef PACKED_BIT_WORD_H_
#define PACKED_BIT_WORD_H_

#include "bit.h"
#include "bit_word.h"
#include "packed_bit_word_slice.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <random>
#include <sstream>

// Pack bit_words, allocated with alignment and padding enabling SIMD
// instructions. Due to padding, the smallest tableaus are 256 bits. For
// performance tableau_element does not store "intendend" size. Only store the
// padded size.
template <size_t word_size> struct packed_bit_word {
  size_t num_bit_words;
  union {
    uint8_t* u8;
    uint64_t* u64;
    bit_word<word_size>* bw;
  };

  explicit packed_bit_word(size_t num_bits)
      : num_bit_words(bits_to_word_padded<word_size>(num_bits)),
        u64(malloc_aligned_padded(num_bits)) {}

  ~packed_bit_word() {
    if (nullptr != u64) {
      bit_word<word_size>::aligned_free(u64);
      u64 = nullptr;
      num_bit_words = 0;
    }
  }

  // copy constructor
  packed_bit_word(const packed_bit_word& other)
      : num_bit_words(other.num_bit_words),
        u64(malloc_aligned_padded(other.num_bit_words * word_size)) {
    memcpy(u8, other.u8, other.num_bit_words * word_size / 8);
  }

  packed_bit_word(const packed_bit_word_slice<word_size>& other)
      : num_bit_words(other.num_bit_words),
        u64(malloc_aligned_padded(other.num_bit_words * word_size)) {
    memcpy(u8, other.u8, other.num_bit_words * word_size / 8);
  }

  // move constructor, is not allowed to throw generally
  packed_bit_word(packed_bit_word&& other) noexcept
      : num_bit_words(other.num_bit_words), u64(other.u64) {
    other.u64 = nullptr;
    other.num_bit_words = 0;
  }

  // copy assignment, deep copy
  packed_bit_word& operator=(const packed_bit_word& other) {
    return *this = packed_bit_word_slice<word_size>(other);
  }

  packed_bit_word<word_size>&
  operator=(const packed_bit_word_slice<word_size>& other) {
    if (num_bit_words == other.num_bit_words) {
      // avoid re-allocating memory
      packed_bit_word_slice<word_size>(*this) = other;
      return *this;
    }

    this->~packed_bit_word();
    new (this) packed_bit_word(other);
    return *this;
  }

  // move assignment
  packed_bit_word& operator=(const packed_bit_word&& other) noexcept {
    this->~packed_bit_word();
    new (this) packed_bit_word(std::move(other));
    return *this;
  }

  // equality
  bool operator==(const packed_bit_word<word_size>& other) const {
    return num_bit_words == other.num_bit_words &&
           memcmp(bw, other.bw, word_size * num_bit_words / 8) == 0;
  }

  bool operator==(const packed_bit_word_slice<word_size>& other) const {
    return num_bit_words == other.num_bit_words &&
           memcmp(bw, other.bw, word_size * num_bit_words / 8) == 0;
  }

  // inequality
  bool operator!=(const packed_bit_word<word_size>& other) const {
    return !(*this == other);
  }

  bool operator!=(const packed_bit_word_slice<word_size>& other) const {
    return !(*this == other);
  }

  // convert packed_bit_word to packed_bit_word_slice
  operator packed_bit_word_slice<word_size>() {
    return packed_bit_word_slice<word_size>(bw, num_bit_words);
  }

  operator const packed_bit_word_slice<word_size>() const {
    return packed_bit_word_slice<word_size>(bw, num_bit_words);
  }

  // index operation
  bit operator[](size_t index) { return bit(u64, index); }
  const bit operator[](size_t index) const { return bit(u64, index); }

  // assignment
  packed_bit_word<word_size>&
  operator^=(const packed_bit_word<word_size>& other) {
    packed_bit_word_slice<word_size>(*this) ^=
        packed_bit_word_slice<word_size>(other);
    return *this;
  }

  packed_bit_word<word_size>
  operator&=(const packed_bit_word<word_size>& other) {
    packed_bit_word_slice<word_size>(*this) &=
        packed_bit_word_slice<word_size>(other);
    return *this;
  }

  packed_bit_word<word_size>
  operator|=(const packed_bit_word<word_size>& other) {
    packed_bit_word_slice<word_size>(*this) |=
        packed_bit_word_slice<word_size>(other);
    return *this;
  }

  packed_bit_word<word_size>&
  operator+=(const packed_bit_word<word_size>& other) {
    size_t num_u64 = (num_bit_words * word_size) >> 6;
    for (size_t i = 0; i < num_u64 - 1; ++i) {
      u64[i] += other.u64[i];
      // carry
      u64[i + 1] += (u64[i] < other.u64[i]);
    }
    u64[num_u64 - 1] += other.u64[num_u64 - 1];
    return *this;
  }

  // compare operation
  bool operator<(const packed_bit_word<word_size>& other) const {
    return packed_bit_word_slice<word_size>(*this) <
           packed_bit_word_slice<word_size>(other);
  }

  // shift operator
  packed_bit_word<word_size>& operator>>=(int offset) {
    packed_bit_word_slice<word_size>(*this) >>= offset;
    return *this;
  }

  packed_bit_word<word_size>& operator<<=(int offset) {
    packed_bit_word_slice<word_size>(*this) <<= offset;
    return *this;
  }

  // convert packed_bit_word to string
  operator std::string() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
  }

  // swap two packed_bit_word
  void swap(packed_bit_word<word_size> other) {
    packed_bit_word_slice<word_size>(*this).swap(
        packed_bit_word_slice<word_size>(other));
  }

  // slice operator, offset and num_bit_words should be the number of bit words
  inline packed_bit_word_slice<word_size> slice(size_t offset,
                                                size_t num_bit_words) {
    return packed_bit_word_slice<word_size>(bw + offset, num_bit_words);
  }

  // slice operator, offset and num_bit_words should be the number of bit words
  inline const packed_bit_word_slice<word_size>
  slice(size_t offset, size_t num_bit_words) const {
    return packed_bit_word_slice<word_size>(bw + offset, num_bit_words);
  }

  // determine the list of bit_word is not all zero
  bool is_not_all_zero() const {
    return packed_bit_word_slice<word_size>(*this).is_not_all_zero();
  }

  // generate random packed_bit_word
  static packed_bit_word<word_size> random(size_t num_bits,
                                           std::mt19937_64& rng) {
    packed_bit_word<word_size> result(num_bits);
    result.randomize(num_bits, rng);
    return result;
  }

  // if num_bits != this.num_bit_words * word_size, this function will change
  // data from low to high
  void randomize(size_t num_bits, std::mt19937_64& rng) {
    auto num_u64 = num_bits >> 6;
    for (size_t k = 0; k < num_u64; ++k)
      u64[k] = rng();

    auto remaining_bits = num_bits & 63;
    if (remaining_bits) {
      auto mask = (uint64_t(1) << remaining_bits) - 1;
      u64[num_u64] &= ~mask;
      u64[num_u64] |= rng() & mask;
    }
  }

  void truncated_overwrite_from(packed_bit_word_slice<word_size>& other,
                                size_t num_bits) {
    size_t num_u8 = num_bits >> 3;
    memcpy(u8, other.u8, num_u8);
    auto remaining_bits = num_bits & 7;
    if (remaining_bits) {
      auto mask = (uint8_t(1) << remaining_bits) - 1;
      u8[num_u8] &= ~mask;
      u8[num_u8] |= other.u8[num_u8] & mask;
    }
  }

  // count the number of 1s
  size_t count() { return packed_bit_word_slice<word_size>(*this).count(); }

  // malloc aligned padded
  uint64_t* malloc_aligned_padded(size_t bits) {
    size_t num_u8 = bits_to_bits_padded<word_size>(bits);
    void* result = bit_word<word_size>::aligned_malloc(num_u8);
    memset(result, 0, num_u8);
    return reinterpret_cast<uint64_t*>(result);
  }
};

// bit operations
template <size_t word_size>
packed_bit_word<word_size>
operator^(const packed_bit_word_slice<word_size>& left,
          const packed_bit_word_slice<word_size>& right) {
  assert(left.num_bit_words == right.num_bit_words);
  packed_bit_word<word_size> result(left.num_bit_words);
  packed_bit_word_slice<word_size>(result).for_each_word(
      left, right, [](auto& a, auto& b, auto& c) { a = b ^ c; });
  return result;
}

template <size_t word_size>
packed_bit_word<word_size> operator^(const packed_bit_word<word_size>& left,
                                     const packed_bit_word<word_size>& right) {
  return packed_bit_word_slice<word_size>(left) ^
         packed_bit_word_slice<word_size>(right);
}

template <size_t word_size>
packed_bit_word<word_size>
operator^(const packed_bit_word<word_size>& left,
          const packed_bit_word_slice<word_size>& right) {
  return packed_bit_word_slice<word_size>(left) ^ right;
}

template <size_t word_size>
packed_bit_word<word_size>
operator^(const packed_bit_word_slice<word_size>& left,
          const packed_bit_word<word_size>& right) {
  return left ^ packed_bit_word_slice<word_size>(right);
}

template <size_t word_size>
packed_bit_word<word_size>
operator&(const packed_bit_word_slice<word_size>& left,
          const packed_bit_word_slice<word_size>& right) {
  assert(left.num_bit_words == right.num_bit_words);
  packed_bit_word<word_size> result(left.num_bit_words);
  packed_bit_word_slice<word_size>(result).for_each_word(
      left, right, [](auto& a, auto& b, auto& c) { a = b & c; });
  return result;
}

template <size_t word_size>
packed_bit_word<word_size> operator&(const packed_bit_word<word_size>& left,
                                     const packed_bit_word<word_size>& right) {
  return packed_bit_word_slice<word_size>(left) &
         packed_bit_word_slice<word_size>(right);
}

template <size_t word_size>
packed_bit_word<word_size>
operator&(const packed_bit_word<word_size>& left,
          const packed_bit_word_slice<word_size>& right) {
  return packed_bit_word_slice<word_size>(left) & right;
}

template <size_t word_size>
packed_bit_word<word_size>
operator&(const packed_bit_word_slice<word_size>& left,
          const packed_bit_word<word_size>& right) {
  return left & packed_bit_word_slice<word_size>(right);
}

template <size_t word_size>
packed_bit_word<word_size>
operator|(const packed_bit_word_slice<word_size>& left,
          const packed_bit_word_slice<word_size>& right) {
  assert(left.num_bit_words == right.num_bit_words);
  packed_bit_word<word_size> result(left.num_bit_words);
  packed_bit_word_slice<word_size>(result).for_each_word(
      left, right, [](auto& a, auto& b, auto& c) { a = b | c; });
  return result;
}

template <size_t word_size>
packed_bit_word<word_size> operator|(const packed_bit_word<word_size>& left,
                                     const packed_bit_word<word_size>& right) {
  return packed_bit_word_slice<word_size>(left) |
         packed_bit_word_slice<word_size>(right);
}

template <size_t word_size>
packed_bit_word<word_size>
operator|(const packed_bit_word<word_size>& left,
          const packed_bit_word_slice<word_size>& right) {
  return packed_bit_word_slice<word_size>(left) | right;
}

template <size_t word_size>
packed_bit_word<word_size>
operator|(const packed_bit_word_slice<word_size>& left,
          const packed_bit_word<word_size>& right) {
  return left | packed_bit_word_slice<word_size>(right);
}

template <size_t word_size>
std::ostream& operator<<(std::ostream& os,
                         const packed_bit_word<word_size> word) {
  for (size_t i = 0; i < word.num_bit_words * word_size; ++i)
    os << "_1"[word[i]];

  return os;
}

#endif
