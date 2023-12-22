#ifndef SIMULATOR_H_
#define SIMULATOR_H_

#include "circuit.h"
#include "span_ref.h"
#include "table.h"
#include "tableau.h"
#include <cstddef>
#include <ios>
#include <ostream>
#include <tuple>

// strore measurement result
struct measurement_record {
  using result = std::tuple<size_t, size_t, bool>;
  std::vector<result> storage;

  void record(size_t qubit, size_t cbit, bool result) {
    storage.push_back({qubit, cbit, result});
  }
  auto operator[](size_t index) { return storage[index]; }
  auto const operator[](size_t index) const { return storage[index]; }
  auto begin() { return storage.begin(); }
  auto end() { return storage.end(); }
  auto begin() const { return storage.begin(); }
  auto end() const { return storage.end(); }
  auto cbegin() const { return storage.cbegin(); }
  auto cend() const { return storage.cend(); }
  auto size() const { return storage.size(); }
  void clear() { storage.clear(); }
};

// quantum circuit simulator, include original tableau, measurement record and a
// random number generator
template <size_t word_size> struct circuit_simulator {

  tableau<word_size> sim_tableau;
  measurement_record sim_record;
  std::mt19937_64 rng;

  // constructor
  // The number of qubits is specified by num_qubits. And the random number
  // generator is used to generate random numbers for error gate and
  // measurement.
  explicit circuit_simulator(size_t num_qubits, size_t seed = 42)
      : sim_tableau(num_qubits), sim_record(), rng() {
    rng.seed(seed);
  }

  // do a quantum gate except measurement gate
  template <size_t vec_size, size_t... S>
  result unpack_vector(auto f, tableau<word_size>& t, auto& vec,
                       std::index_sequence<S...>) {

    // TODO: maybe can be better
    return std::get<vec_size - 1>(f)(t, vec[S]...);
  }

  template <size_t vec_size>
  result unpack_vector(auto f, tableau<word_size>& t, auto& vec) {
    if (vec.size() != vec_size)
      throw std::runtime_error("wrong number of qubits for gate");
    return unpack_vector<vec_size>(f, t, vec,
                                   std::make_index_sequence<vec_size>());
  }

  // do a measurement gate
  template <size_t... S>
  result unpack_vector(auto f, tableau<word_size>& t, auto& vec,
                       std::mt19937_64& rng, std::index_sequence<S...>) {

    // TODO: maybe can be better
    return std::get<gate_type::COLLAPSING_GATE>(f)(t, rng, vec[S]...);
  }

  template <size_t vec_size>
  result unpack_vector(auto f, tableau<word_size>& t, auto& vec,
                       std::mt19937_64& rng) {
    if (vec.size() != vec_size)
      throw std::runtime_error("wrong number of qubits for gate");
    return unpack_vector(f, t, vec, rng, std::make_index_sequence<vec_size>());
  }

  // do a quantum circuit, check the max qubit of the quantum circuit, if it is
  // larger than the number of qubits of the tableau, expand the tableau
  void do_circuit(const quantum_circuit& qc) {
    if (qc.max_qubit() >= sim_tableau.num_qubits) {
      sim_tableau.expand(qc.max_qubit(), 1.0);
      std::cerr << "WARNING: expanding tableau to " << qc.max_qubit()
                << " qubits\n";
    }

    // according quantum circuit instruction type to do the instruction
    qc.for_each_circuit_instruction([&](const circuit_instruction& ci) {
      if (!gate_map<word_size>.contains(ci.gate + "_gate")) {
        throw std::runtime_error("unknown gate");
      }

      auto pair = gate_map<word_size>[ci.gate + "_gate"];
      if (SINGLE_QUBIT_GATE == pair.first) {
        unpack_vector<1>(pair.second, sim_tableau, ci.targets);
      } else if (TWO_QUBIT_GATE == pair.first) {
        unpack_vector<2>(pair.second, sim_tableau, ci.targets);
      } else if (COLLAPSING_GATE == pair.first) {
        auto record =
            unpack_vector<1>(pair.second, sim_tableau, ci.targets, rng);
        if (record.has_value())
          sim_record.record(ci.targets[0], static_cast<size_t>(ci.args[0]),
                            record.value());
      } else if (ERROR_QUBIT_GATE == pair.first) {
        std::bernoulli_distribution d(ci.args[0]);
        if (d(rng))
          unpack_vector<1>(pair.second, sim_tableau, ci.targets);
      } else {
        throw std::runtime_error("unknown gate");
      }
    });
  }

  // do a quantum circuit instruction
  void do_circuit_instruction(const circuit_instruction& ci) {

    if (!gate_map<word_size>.contains(ci.gate + "_gate")) {
      throw std::runtime_error("unknown gate");
    }

    auto pair = gate_map<word_size>[ci.gate + "_gate"];

    if (SINGLE_QUBIT_GATE == pair.first) {
      unpack_vector<1>(pair.second, sim_tableau, ci.targets);
    } else if (TWO_QUBIT_GATE == pair.first) {
      unpack_vector<2>(pair.second, sim_tableau, ci.targets);
    } else if (COLLAPSING_GATE == pair.first) {
      auto record = unpack_vector<1>(pair.second, sim_tableau, ci.targets, rng);
      if (record.has_value())
        sim_record.record(ci.targets[0], static_cast<size_t>(ci.args[0]),
                          record.value());
    } else if (ERROR_QUBIT_GATE == pair.first) {
      std::bernoulli_distribution d(ci.args[0]);
      if (d(rng))
        unpack_vector<1>(pair.second, sim_tableau, ci.targets);
    } else {
      throw std::runtime_error("unknown gate");
    }
  }

  // sample the quantum circuit, after each iteration, reset the tableau to
  // identity
  void sample(const quantum_circuit& qc, size_t num_samples) {
    for (size_t i = 0; i < num_samples; i++) {
      reset_tableau();
      do_circuit(qc);
    }
  }

  // reset the tableau to identity
  void reset_tableau() { sim_tableau.reset(); }

  // z-basis reset
  void reset(size_t qubit) { sim_tableau.reset(rng, qubit); }

  // // x-basis reset
  // void reset_x(size_t qubit) { sim_tableau.reset_x(rng, qubit); }

  // // y-basis reset
  // void reset_y(size_t qubit) { sim_tableau.reset_y(rng, qubit); }

  // measurement record size
  size_t record_size() { return sim_record.size(); }

  // get the measurement record
  auto current_measurement_record() const { return sim_record; }

  auto measure_all() {
    measurement_record mr;
    for (size_t i = 0; i < sim_tableau.num_qubits; i++) {
      auto record = sim_tableau.m_gate(rng, i);
      if (record.has_value())
        mr.record(record.value());
    }
    return mr;
  }
};

#endif
