#ifndef SPAN_REF_H_
#define SPAN_REF_H_

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

// A significant distinction between the semantics of this class and the
// std::span class introduced in C++20 is that this class defines equality and
// ordering operators based on *the content being pointed to* rather than the
// values of the pointers themselves. Two range references are not considered
// equal simply because they have identical pointers; they are deemed equal
// because they point to ranges with matching contents. In essence, this class
// behaves more like a *reference* rather than a pointer.

template <typename T> struct span_ref {
  T* ptr_start;
  T* ptr_end;

  span_ref() : ptr_start(nullptr), ptr_end(nullptr) {}
  span_ref(T* begin, T* end) : ptr_start(begin), ptr_end(end) {}

  // Implicit conversions.
  span_ref(T* singleton) : ptr_start(singleton), ptr_end(singleton + 1) {}

  span_ref(const span_ref<typename std::remove_const<T>::type>& other)
      : ptr_start(other.ptr_start), ptr_end(other.ptr_end) {}

  span_ref(std::vector<T>& items)
      : ptr_start(items.data()), ptr_end(items.data() + items.size()) {}

  span_ref(const std::vector<typename std::remove_const<T>::type>& items)
      : ptr_start(items.data()), ptr_end(items.data() + items.size()) {}

  template <size_t K>
  span_ref(std::array<T, K>& items)
      : ptr_start(items.data()), ptr_end(items.data() + items.size()) {}

  template <size_t K>
  span_ref(const std::array<typename std::remove_const<T>::type, K>& items)
      : ptr_start(items.data()), ptr_end(items.data() + items.size()) {}

  span_ref sub(size_t start_offset, size_t end_offset) const {
    return span_ref<T>(ptr_start + start_offset, ptr_start + end_offset);
  }

  size_t size() const { return ptr_end - ptr_start; }

  const T* begin() const { return ptr_start; }

  const T* end() const { return ptr_end; }

  const T& back() const { return *(ptr_end - 1); }

  const T& front() const { return *ptr_start; }

  bool empty() const { return ptr_end == ptr_start; }

  T* begin() { return ptr_start; }

  T* end() { return ptr_end; }

  T& back() { return *(ptr_end - 1); }

  T& front() { return *ptr_start; }

  const T& operator[](size_t index) const { return ptr_start[index]; }

  T& operator[](size_t index) { return ptr_start[index]; }

  bool operator==(const span_ref<const T>& other) const {
    size_t n = size();
    if (n != other.size()) {
      return false;
    }
    for (size_t k = 0; k < n; k++) {
      if (ptr_start[k] != other[k]) {
        return false;
      }
    }
    return true;
  }

  bool
  operator==(const span_ref<typename std::remove_const<T>::type>& other) const {
    return span_ref<const T>(ptr_start, ptr_end) ==
           span_ref<const T>(other.ptr_start, other.ptr_end);
  }

  bool operator!=(const span_ref<const T>& other) const {
    return !(*this == other);
  }

  bool
  operator!=(const span_ref<typename std::remove_const<T>::type>& other) const {
    return !(*this == other);
  }

  std::string str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
  }

  // Lexicographic ordering.
  bool operator<(const span_ref<const T>& other) const {
    auto n = std::min(size(), other.size());
    for (size_t k = 0; k < n; k++) {
      if ((*this)[k] != other[k]) {
        return (*this)[k] < other[k];
      }
    }
    return size() < other.size();
  }

  bool
  operator<(const span_ref<typename std::remove_const<T>::type>& other) const {
    return span_ref<const T>(ptr_start, ptr_end) <
           span_ref<const T>(other.ptr_start, other.ptr_end);
  }
};

// Wraps an iterable object so that its values are printed with comma
// separators.
template <typename t_iter> struct seperator;

/// A wrapper indicating a range of values should be printed with comma
/// separators.
template <typename t_iter> struct seperator {
  const t_iter& iter;
  const char* sep;
  std::string str() const {
    std::stringstream out;
    out << *this;
    return out.str();
  }
};

template <typename t_iter>
seperator<t_iter> seperate(const t_iter& v, const char* sep = ", ") {
  return seperator<t_iter>{v, sep};
}

template <typename t_iter>
std::ostream& operator<<(std::ostream& out, const seperator<t_iter>& v) {
  bool first = true;
  for (const auto& t : v.iter) {
    if (first) {
      first = false;
    } else {
      out << v.sep;
    }
    out << t;
  }
  return out;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const span_ref<T>& v) {
  out << "span_ref{" << seperate(v) << "}";
  return out;
}

// A memory resource that efficiently accumulates data incrementally.

// There are three important types of "region" involved: the tail region, the
// current region, and old regions.

// The tail is used for adding contiguous data to the buffer in an incremental
// manner. When the tail exceeds the available storage, more memory is allocated
// and the tail is copied into it to maintain contiguity. The tail can be
// discarded or committed at any time. Discarding allows reusing the covered
// memory when writing the next tail. Committing permanently preserves the data
// (until clearing or deconstructing the monotonic buffer) and ensures that it
// will not move so that pointers to it can be stored.

// The current region is a continuous block of memory where the tail is being
// written. If the tail grows beyond this region and triggers an allocation,
// then this current region becomes an old region, while newly allocated memory
// becomes the new current region. Each subsequent current region will have at
// least double size compared to its predecessor.

// Old regions are finalized memory segments that will be retained until
// clearing or deconstructing of the buffer occurs.

// some ref: https://zhuanlan.zhihu.com/p/96089089
template <typename T> struct monotonic_buffer {

  // Contiguous memory that is being appended to, but has not yet been
  // committed.
  span_ref<T> tail;

  // The current contiguous memory region with a mix of committed, staged, and
  // unused memory.
  span_ref<T> cur;

  // Old contiguous memory regions that have been committed and now need to be
  // kept.
  std::vector<span_ref<T>> old_areas;

  // Constructs an empty monotonic buffer.
  monotonic_buffer() : tail(), cur(), old_areas() {}

  // Constructs an empty monotonic buffer with initial capacity for its current
  // region.
  monotonic_buffer(size_t reserve) { ensure_available(reserve); }

  void _soft_clear() {
    cur.ptr_start = nullptr;
    cur.ptr_end = nullptr;
    tail.ptr_start = nullptr;
    tail.ptr_end = nullptr;
    old_areas.clear();
  }

  void _hard_clear() {
    for (auto old : old_areas) {
      free(old.ptr_start);
    }
    if (nullptr != cur.ptr_start) {
      free(cur.ptr_start);
    }
  }

  ~monotonic_buffer() { _hard_clear(); }

  monotonic_buffer(monotonic_buffer&& other) noexcept
      : tail(other.tail), cur(other.cur),
        old_areas(std::move(other.old_areas)) {
    other._soft_clear();
  }

  monotonic_buffer(const monotonic_buffer& other) = delete;

  monotonic_buffer& operator=(monotonic_buffer&& other) noexcept {
    _hard_clear();
    cur = other.cur;
    tail = other.tail;
    old_areas = std::move(other.old_areas);
    other._soft_clear();
    return *this;
  }

  // Invalidates all previous data and resets the class into a clean state.
  // Happens to keep the current contiguous memory region and free old regions.
  void clear() {
    for (auto old : old_areas) {
      free(old.ptr_start);
    }
    old_areas.clear();
    tail.ptr_end = tail.ptr_start = cur.ptr_start;
  }

  // Returns the size of memory allocated and held by this monotonic buffer (in
  // units of sizeof(T)).
  size_t total_allocated() const {
    size_t result = cur.size();
    for (auto old : old_areas) {
      result += old.size();
    }
    return result;
  }

  // Appends and commits data.
  // Requires the tail to be empty, to avoid bugs where previously staged data
  // is committed.
  span_ref<T> take_copy(span_ref<const T> data) {
    assert(tail.size() == 0);
    append_tail(data);
    return commit_tail();
  }

  // Adds a staged data item.
  void append_tail(T item) {
    ensure_available(1);
    *tail.ptr_end = item;
    tail.ptr_end++;
  }

  // Adds staged data.
  void append_tail(span_ref<const T> data) {
    ensure_available(data.size());
    std::copy(data.begin(), data.end(), tail.ptr_end);
    tail.ptr_end += data.size();
  }

  // Throws away staged data, so its memory can be re-used.
  void discard_tail() { tail.ptr_end = tail.ptr_start; }

  // Changes staged data into committed data that will be kept until the buffer
  // is cleared or deconstructed.
  span_ref<T> commit_tail() {
    span_ref<T> result(tail);
    tail.ptr_start = tail.ptr_end;
    return result;
  }

  // Ensures it is possible to stage at least `min_required` more items without
  // more reallocations.
  void ensure_available(size_t min_required) {
    size_t available = cur.ptr_end - tail.ptr_end;
    if (available >= min_required) {
      return;
    }

    size_t alloc_count = std::max(min_required, cur.size() << 1);
    if (nullptr != cur.ptr_start) {
      old_areas.push_back(cur);
    }
    cur.ptr_start = (T*)malloc(alloc_count * sizeof(T));
    cur.ptr_end = cur.ptr_start + alloc_count;

    // Staged data is not complete yet; keep it contiguous by copying it to the
    // new larger memory region.
    size_t tail_size = tail.size();
    if (tail_size) {
      std::move(tail.ptr_start, tail.ptr_end, cur.ptr_start);
    }

    tail = {cur.ptr_start, cur.ptr_start + tail_size};
  }
};

#endif
