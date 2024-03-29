#include <functional>

#ifdef FUNCTION_REGISTRATION
#define SINGLE_QUBIT_GATE(GATE_NAME, ...)                                      \
  template <size_t word_size>                                                  \
  result GATE_NAME##_gate(tableau<word_size>& t, const size_t qubit)           \
      __VA_ARGS__

#define TWO_QUBIT_GATE(GATE_NAME, ...)                                         \
  template <size_t word_size>                                                  \
  result GATE_NAME##_gate(tableau<word_size>& t, const size_t qubit1,          \
                          const size_t qubit2) __VA_ARGS__

#define COLLAPSING_GATE(GATE_NAME, ...)                                        \
  template <size_t word_size>                                                  \
  result GATE_NAME##_gate(tableau<word_size>& t, std::mt19937_64& rng,         \
                          const size_t qubit) __VA_ARGS__

#define ERROR_QUBIT_GATE(GATE_NAME, ...)                                       \
  template <size_t word_size>                                                  \
  result GATE_NAME##_gate(tableau<word_size>& t, const size_t qubit)           \
      __VA_ARGS__
#endif

#ifdef STRUCT_FUNCTION_REGISTRATION
#define SINGLE_QUBIT_GATE(GATE_NAME, ...)                                      \
  result GATE_NAME##_gate(const size_t qubit) {                                \
    tableau<word_size>& t = *this;                                             \
    __VA_ARGS__                                                                \
  }

#define TWO_QUBIT_GATE(GATE_NAME, ...)                                         \
  result GATE_NAME##_gate(const size_t qubit1, const size_t qubit2) {          \
    tableau<word_size>& t = *this;                                             \
    __VA_ARGS__                                                                \
  }

#define COLLAPSING_GATE(GATE_NAME, ...)                                        \
  result GATE_NAME##_gate(std::mt19937_64& rng, const size_t qubit) {          \
    tableau<word_size>& t = *this;                                             \
    __VA_ARGS__                                                                \
  }

#define ERROR_QUBIT_GATE(GATE_NAME, ...)                                       \
  result GATE_NAME##_gate(const size_t qubit) {                                \
    tableau<word_size>& t = *this;                                             \
    __VA_ARGS__                                                                \
  }
#endif

#ifdef GATE_MAP_REGISTRATION
#define SINGLE_QUBIT_GATE(GATE_NAME, ...)                                      \
  {STRINGIZE(GATE_NAME##_gate),                                                \
             {SINGLE_QUBIT_GATE, GATE_NAME##_gate<word_size>}},

#define TWO_QUBIT_GATE(GATE_NAME, ...)                                         \
  {STRINGIZE(GATE_NAME##_gate), {TWO_QUBIT_GATE, GATE_NAME##_gate<word_size>}},

#define COLLAPSING_GATE(GATE_NAME, ...)                                        \
  {STRINGIZE(GATE_NAME##_gate), {COLLAPSING_GATE, GATE_NAME##_gate<word_size>}},

#define ERROR_QUBIT_GATE(GATE_NAME, ...)                                       \
  {STRINGIZE(GATE_NAME##_gate),                                                \
             {ERROR_QUBIT_GATE, GATE_NAME##_gate<word_size>}},
#endif
