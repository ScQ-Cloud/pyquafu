#ifndef BIT_H_
#define BIT_H_

#include <cstddef>
#include <cstdint>

// bit in byte
struct bit {
  uint8_t* byte;
  uint8_t byte_index;

  bit(void* ptr, size_t offset)
      : byte(((uint8_t*)ptr + (offset / 8))), byte_index(offset & 7) {}

  // copy assignment for bit in byte
  inline bit& operator=(bool value) {
    // make bit be 0
    *byte &= ~((uint8_t)1 << byte_index);
    // assignment
    *byte |= uint8_t(value) << byte_index;
    return *this;
  }

  inline bit& operator=(const bit& other) {
    *this = bool(other);
    return *this;
  }

  // bit operator
  inline bit& operator^=(bool value) {
    *byte ^= uint8_t(value) << byte_index;
    return *this;
  }

  inline bit& operator&=(bool value) {
    *byte &= (uint8_t(value) << byte_index) | ~(uint8_t(1) << byte_index);
    return *this;
  }

  inline bit& operator|=(bool value) {
    *byte |= uint8_t(value) << byte_index;
    return *this;
  }

  // conversion operator
  inline operator bool() const { return (*byte >> byte_index) & 1; }

  void swap(bit other) {
    bool b = bool(other);
    other = bool(*this);
    *this = b;
  }
};

#endif
