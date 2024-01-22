#ifndef CIRCUIT_H_
#define CIRCUIT_H_

#include "span_ref.h"
#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

// gate instruction, include gate name, target qubits and arguments
struct circuit_instruction {
  std::string gate;
  span_ref<const size_t> targets;
  span_ref<const double> args;

  circuit_instruction() = delete;
  circuit_instruction(std::string gate, span_ref<const size_t> targets,
                      span_ref<const double> args = {})
      : gate(gate), targets(targets), args(args) {}

  // TODO: add validation for instruction
  void validate() {}
};

inline std::ostream& operator<<(std::ostream& os,
                                const circuit_instruction& instr) {
  os << instr.gate << " "
     << "targets: ";
  for (auto& target : instr.targets) {
    os << target << " ";
  }

  if (instr.args.size() > 0) {
    os << "args: ";
    for (auto& arg : instr.args) {
      os << arg << " ";
    }
  }
  os << "\n";
  return os;
}

// quantum circuit, include a list of gate instructions, and buffer for targets
// and args
struct quantum_circuit {
  std::vector<circuit_instruction> instr_list;
  monotonic_buffer<size_t> targets_buf;
  monotonic_buffer<double> args_buf;

  quantum_circuit() : instr_list(), targets_buf(), args_buf() {}
  quantum_circuit(const std::string& gate, const std::vector<size_t>& targets,
                  const std::vector<double>& args = {})
      : instr_list(), targets_buf(), args_buf() {
    _append(gate, targets, args);
  }

  // copy constructor
  quantum_circuit(const quantum_circuit& other)
      : instr_list(other.instr_list),
        targets_buf(other.targets_buf.total_allocated()),
        args_buf(other.args_buf.total_allocated()) {

    // take copy of targets and args
    for (auto& instr : instr_list) {
      instr.targets = targets_buf.take_copy(instr.targets);
      instr.args = args_buf.take_copy(instr.args);
    }
  };

  // move constructor
  quantum_circuit(quantum_circuit&& other) noexcept
      : instr_list(std::move(other.instr_list)),
        targets_buf(std::move(other.targets_buf)),
        args_buf(std::move(other.args_buf)) {}

  // copy assignment
  quantum_circuit& operator=(const quantum_circuit& other) {
    instr_list = other.instr_list;
    targets_buf = monotonic_buffer<size_t>(other.targets_buf.total_allocated());
    args_buf = monotonic_buffer<double>(other.args_buf.total_allocated());
    for (auto& instr : instr_list) {
      instr.targets = targets_buf.take_copy(instr.targets);
      instr.args = args_buf.take_copy(instr.args);
    }
    return *this;
  }

  // move assignment
  quantum_circuit& operator=(quantum_circuit&& other) noexcept {
    instr_list = std::move(other.instr_list);
    targets_buf = std::move(other.targets_buf);
    args_buf = std::move(other.args_buf);
    return *this;
  }

  // concatenate two quantum circuits
  quantum_circuit& operator+=(const quantum_circuit& other) {
    span_ref<const circuit_instruction> other_instrs(other.instr_list);

    if (&other == this) {
      instr_list.insert(instr_list.end(), other_instrs.begin(),
                        other_instrs.end());
      return *this;
    }

    for (auto& instr : other_instrs) {
      auto instr_targets = targets_buf.take_copy(instr.targets);
      auto instr_args = args_buf.take_copy(instr.args);
      instr_list.push_back({instr.gate, instr_targets, instr_args});
    }

    return *this;
  }

  // concatenate two quantum circuits
  quantum_circuit operator+(const quantum_circuit& other) {
    quantum_circuit result(*this);
    result += other;
    return result;
  }

  friend std::ostream& operator<<(std::ostream& os, const quantum_circuit& qc) {
    for (auto& instr : qc.instr_list) {
      os << instr;
    }
    return os;
  }

  operator std::string() const { return str(); }

  std::string str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
  }

  quantum_circuit& append(const std::string& gate,
                          const std::vector<size_t>& targets,
                          const std::vector<double>& args = {}) {

    _append(gate, targets, args);

    return *this;
  }

  quantum_circuit& _append(const std::string& gate,
                           span_ref<const size_t> targets,
                           span_ref<const double> args) {
    circuit_instruction instr(gate, targets, args);
    instr.validate();

    // TODO: fuse some instr
    instr.targets = targets_buf.take_copy(targets);
    instr.args = args_buf.take_copy(args);
    instr_list.push_back(instr);

    return *this;
  }

  size_t max_qubit() const {
    size_t max_qubit = 0;
    for (auto& instr : instr_list) {
      for (auto& target : instr.targets) {
        max_qubit = std::max(max_qubit, target);
      }
    }

    return max_qubit;
  }

  template <typename func>
  void for_each_circuit_instruction(const func& callback) const {
    for (auto& instr : instr_list) {
      callback(instr);
    }
  }
};

#endif
