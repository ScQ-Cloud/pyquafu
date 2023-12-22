#ifndef PACKED_BIT_WORD_SLICE_H_
#define PACKED_BIT_WORD_SLICE_H_

#include "bit.h"
#include "bit_word.h"
#include "utils.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <random>
#include <sstream>

// reference to a slice of a packed bit word
template <size_t word_size> struct packed_bit_word_slice {
  const size_t num_bit_words;

  union {
    uint8_t* u8;
    uint64_t* u64;
    bit_word<word_size>* bw;
  };

  packed_bit_word_slice(bit_word<word_size>* bw, size_t num_bit_words)
      : num_bit_words(num_bit_words), bw(bw) {}

  // assignment operator
  packed_bit_word_slice<word_size>&
  operator=(const packed_bit_word_slice<word_size>& other) {
    memcpy(u8, other.u8, num_bit_words * word_size / 8);
    return *this;
  }

  packed_bit_word_slice<word_size>&
  operator^=(const packed_bit_word_slice<word_size>& other) {
    for_each_word(other, [](auto& a, auto& b) { a ^= b; });
    return *this;
  }

  packed_bit_word_slice<word_size>&
  operator&=(const packed_bit_word_slice<word_size>& other) {
    for_each_word(other, [](auto& a, auto& b) { a &= b; });
    return *this;
  }

  packed_bit_word_slice<word_size>&
  operator|=(const packed_bit_word_slice<word_size>& other) {
    for_each_word(other, [](auto& a, auto& b) { a |= b; });
    return *this;
  }

  packed_bit_word_slice<word_size>&
  operator+=(const packed_bit_word_slice<word_size>& other) {
    size_t num_u64 = (num_bit_words * word_size) >> 6;
    for (size_t i = 0; i < num_u64 - 1; ++i) {
      u64[i] += other.u64[i];
      // carry
      u64[i + 1] += (u64[i] < other.u64[i]);
    }
    u64[num_u64 - 1] += other.u64[num_u64 - 1];
    return *this;
  }

  packed_bit_word_slice<word_size>& operator>>=(int offset) {
    uint64_t incoming_word;
    uint64_t current_word;

    if (0 == offset)
      return *this;

    // move right every u64 by offset
    while (64 <= offset) {
      incoming_word = 0;
      for (int w = ((num_bit_words * word_size) >> 6) - 1; w >= 0; w--) {
        current_word = u64[w];
        u64[w] = incoming_word;
        incoming_word = current_word;
      }
      offset -= 64;
    }

    if (0 == offset)
      return *this;

    incoming_word = 0;
    for (int w = ((num_bit_words * word_size) >> 6) - 1; w >= 0; w--) {
      current_word = u64[w];
      // move right
      u64[w] >>= offset;
      // add high bits
      u64[w] |= incoming_word << (64 - offset);
      // update next incoming word
      incoming_word = current_word & ((uint64_t(1) << offset) - 1);
    }

    return *this;
  }

  packed_bit_word_slice<word_size>& operator<<=(int offset) {
    uint64_t incoming_word;
    uint64_t current_word;

    if (0 == offset)
      return *this;

    // move left every u64 by offset
    while (64 <= offset) {
      incoming_word = 0;
      for (uint64_t w = 0; w < (num_bit_words * word_size) >> 6; w++) {
        current_word = u64[w];
        u64[w] = incoming_word;
        incoming_word = current_word;
      }
      offset -= 64;
    }

    if (0 == offset)
      return *this;

    incoming_word = 0;
    for (uint64_t w = 0; w < (num_bit_words * word_size) >> 6; w++) {
      current_word = u64[w];
      // move left
      u64[w] <<= offset;
      // add low bits
      u64[w] |= incoming_word;
      // update next incoming word
      incoming_word = current_word >> (64 - offset);
    }

    return *this;
  }

  // equality operator
  bool operator==(const packed_bit_word_slice<word_size>& other) const {
    return num_bit_words == other.num_bit_words &&
           0 == memcmp(bw, other.bw, num_bit_words * word_size / 8);
  }

  bool operator!=(const packed_bit_word_slice<word_size>& other) const {
    return !(*this == other);
  }

  // index operator
  inline bit operator[](size_t index) { return bit(u8, index); }

  inline const bit operator[](size_t index) const { return bit(u8, index); }

  // convert packed bit word slice to string
  operator std::string() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
  }

  void swap(packed_bit_word_slice<word_size> other) {
    for_each_word(other, [](auto& a, auto& b) { std::swap(a, b); });
  }

  // slice operator, offset and num_bit_words should be the number of bit words
  inline packed_bit_word_slice<word_size> slice(size_t offset,
                                                size_t num_bit_words) {
    return packed_bit_word_slice<word_size>(bw + offset, num_bit_words);
  }

  inline const packed_bit_word_slice<word_size>
  slice(size_t offset, size_t num_bit_words) const {
    return packed_bit_word_slice<word_size>(bw + offset, num_bit_words);
  }

  // determine the list of bit_word is not all zero
  bool is_not_all_zero() const {
    bit_word<word_size> res{};
    for_each_word([&res](auto& a) { res |= a; });
    return bool(res);
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

  // write the truncated data from other to this
  void truncated_overwrite_from(packed_bit_word_slice<word_size> other,
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
  template <size_t W> size_t count() const {
    auto end = u64 + (num_bit_words * word_size >> 6);
    size_t result = 0;
    for (const uint64_t* p = u64; p != end; p++) {
      result += count_uint64_bits(*p);
    }
    return result;
  }

  template <typename func> inline void for_each_word(func f) const {
    auto* bw_start = bw;
    auto* bw_end = bw + num_bit_words;
    while (bw_start != bw_end) {
      f(*bw_start);
      ++bw_start;
    }
  }

  template <typename func>
  inline void for_each_word(packed_bit_word_slice<word_size> other,
                            func f) const {
    auto* bw_start = bw;
    auto* bw_end = bw + num_bit_words;
    auto* other_bw_start = other.bw;
    while (bw_start != bw_end) {
      f(*bw_start, *other_bw_start);
      ++bw_start;
      ++other_bw_start;
    }
  }

  template <typename func>
  inline void for_each_word(packed_bit_word_slice<word_size> other1,
                            packed_bit_word_slice<word_size> other2,
                            func f) const {
    auto* bw_start = bw;
    auto* bw_end = bw + num_bit_words;
    auto* other1_bw_start = other1.bw;
    auto* other2_bw_start = other2.bw;
    while (bw_start != bw_end) {
      f(*bw_start, *other1_bw_start, *other2_bw_start);
      ++bw_start;
      ++other1_bw_start;
      ++other2_bw_start;
    }
  }

  template <typename func>
  inline void for_each_word(packed_bit_word_slice<word_size> other1,
                            packed_bit_word_slice<word_size> other2,
                            packed_bit_word_slice<word_size> other3,
                            func f) const {
    auto* bw_start = bw;
    auto* bw_end = bw + num_bit_words;
    auto* other1_bw_start = other1.bw;
    auto* other2_bw_start = other2.bw;
    auto* other3_bw_start = other3.bw;
    while (bw_start != bw_end) {
      f(*bw_start, *other1_bw_start, *other2_bw_start, *other3_bw_start);
      ++bw_start;
      ++other1_bw_start;
      ++other2_bw_start;
      ++other3_bw_start;
    }
  }

  template <typename func>
  inline void for_each_word(packed_bit_word_slice<word_size> other1,
                            packed_bit_word_slice<word_size> other2,
                            packed_bit_word_slice<word_size> other3,
                            packed_bit_word_slice<word_size> other4,
                            func f) const {
    auto* bw_start = bw;
    auto* bw_end = bw + num_bit_words;
    auto* other1_bw_start = other1.bw;
    auto* other2_bw_start = other2.bw;
    auto* other3_bw_start = other3.bw;
    auto* other4_bw_start = other4.bw;
    while (bw_start != bw_end) {
      f(*bw_start, *other1_bw_start, *other2_bw_start, *other3_bw_start,
        *other4_bw_start);
      ++bw_start;
      ++other1_bw_start;
      ++other2_bw_start;
      ++other3_bw_start;
      ++other4_bw_start;
    }
  }
};

// output operator for packed bit word slice
template <size_t word_size>
std::ostream& operator<<(std::ostream& os,
                         const packed_bit_word_slice<word_size>& word) {
  for (size_t i = 0; i < word.num_bit_words * word_size; ++i)
    os << "_1"[word[i]];

  return os;
}

#endif
