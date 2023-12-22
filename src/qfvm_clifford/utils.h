#ifndef TABLEAU_ELEMENT_UTILS_H_
#define TABLEAU_ELEMENT_UTILS_H_

#include <cstddef>
#include <cstdint>

template <size_t word_size> constexpr size_t bits_to_bits_padded(size_t bits) {
  return (bits + (word_size - 1)) & ~(word_size - 1);
}

template <size_t word_size> constexpr size_t bits_to_word_padded(size_t bits) {
  return bits_to_bits_padded<word_size>(bits) / word_size;
}

// reference:
// https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
inline uint8_t count_uint64_bits(uint64_t value) {
  value = value - ((value >> 1) & 0x5555555555555555ull);
  value =
      (value & 0x3333333333333333ull) + ((value >> 2) & 0x3333333333333333ull);
  return (((value + (value >> 4)) & 0xF0F0F0F0F0F0F0Full) *
          0x101010101010101ull) >>
         56;
}

// Concatenate preprocessor tokens A and B without expanding macro definitions
// (however, if invoked from a macro, macro arguments are expanded).
#define PPCAT_NX(A, B) A##B

// Concatenate preprocessor tokens A and B after macro-expanding them.
#define PPCAT(A, B) PPCAT_NX(A, B)

// Turn A into a string literal without expanding macro definitions (however, if
// invoked from a macro, macro arguments are expanded).
#define STRINGIZE_NX(A) #A

// Turn A into a string literal after macro-expanding it.
#define STRINGIZE(A) STRINGIZE_NX(A)

#endif
