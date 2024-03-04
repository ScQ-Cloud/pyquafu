#include "simulator.hpp"
#include "instructions.hpp"
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <random>
#include <tuple>

#ifdef _USE_GPU
#include <cuda_simulator.cuh>
#endif

#ifdef _USE_CUQUANTUM
#include <custate_simu.cuh>
#endif

#ifdef USE_SIMD
constexpr size_t _word_size = 256;
#else
constexpr size_t _word_size = 64;
#endif

namespace py = pybind11;

template <typename T>
py::array_t<T> to_numpy(const std::tuple<T*, size_t>& src) {
  auto src_ptr = std::get<0>(src);
  auto src_size = std::get<1>(src);

  auto capsule =
      py::capsule(src_ptr, [](void* p) { delete[] reinterpret_cast<T*>(p); });
  return py::array_t<T>(src_size, src_ptr, capsule);
}

py::object applyop_statevec(py::object const& pyop, py::array_t<complex<double>> &np_inputstate){
    py::buffer_info buf = np_inputstate.request();
    auto* data_ptr = reinterpret_cast<std::complex<double>*>(buf.ptr);
    size_t data_size = buf.size;

    auto op =  from_pyops(pyop);
    if (data_size == 0){
        return np_inputstate;
    }
    else{
        StateVector<double> state(data_ptr, buf.size);
        apply_op(*op, state);
        state.move_data_to_python();
        return np_inputstate;
    }
}

py::dict sampling_statevec(py::dict const& pymeas, py::array_t<complex<double>> &np_inputstate, int shots){
    std::vector<std::pair<uint, uint> > measures;
    for (auto item : pymeas){
        int qbit = item.first.cast<uint>();
        int cbit = item.second.cast<uint>();
        measures.push_back(std::pair<uint, uint>(qbit, cbit));
    }
    py::buffer_info buf = np_inputstate.request();
    auto* data_ptr = reinterpret_cast<std::complex<double>*>(buf.ptr);
    size_t data_size = buf.size;
    StateVector<double> state(data_ptr, buf.size);
    auto counts = state.measure_samples(measures, shots);
    state.move_data_to_python();
    return py::cast(counts);
}

std::pair<std::map<uint, uint>, py::array_t<complex<double>>>
simulate_circuit(py::object const& pycircuit,
                 py::array_t<complex<double>>& np_inputstate,
                 const int& shots) {
    auto circuit = Circuit(pycircuit);
    py::buffer_info buf = np_inputstate.request();
    auto* data_ptr = reinterpret_cast<std::complex<double>*>(buf.ptr);
    size_t data_size = buf.size;
    // If measure all at the end, simulate once
    uint actual_shots = shots;
    if (circuit.final_measure())
        actual_shots = 1;
    vector<std::pair<uint, uint>> measures = circuit.measure_vec();
    std::map<uint, bool> cbit_measured;
    for (auto& pair : measures) {
        cbit_measured[pair.second] = true;
    }

    // Store outcome's count
    std::map<uint, uint> outcount;
    StateVector<double> state;
    if (data_size != 0){
        state.load_data(data_ptr, data_size);
    }
    if (circuit.final_measure()){
        simulate(circuit, state);
        if (!measures.empty()){
            auto countstr = state.measure_samples(circuit.measure_vec(), shots);
            for (auto it : countstr){
                uint si = std::stoi(it.first, nullptr, 2);
                outcount[si] = it.second;
            }
        }
        if (data_size == 0)
            return std::make_pair(outcount,
                            to_numpy(state.move_data_to_python()));
        else
            state.move_data_to_python();
            return std::make_pair(outcount, np_inputstate);
    }
    else{
        for (uint i = 0; i < shots; i++) {
            StateVector<double> buffer;
            if (data_size != 0){
                auto buffer_ptr = std::make_unique<complex<double>[]>(data_size);
                std::copy(data_ptr, data_ptr + data_size, buffer_ptr.get());
                buffer.load_data(buffer_ptr, data_size);
            }else{
                buffer = StateVector<double>();
            }
            simulate(circuit, buffer);
            // store reg
            vector<uint> tmpcreg = buffer.creg();
            uint outcome = 0;
            for (uint j = 0; j < tmpcreg.size(); j++) {
                if (cbit_measured.find(j) == cbit_measured.end())
                continue;
                outcome *= 2;
                outcome += tmpcreg[j];
            }
            if (outcount.find(outcome) != outcount.end())
                outcount[outcome]++;
            else
                outcount[outcome] = 1;

            if (i == shots-1){
                return std::make_pair(outcount,
                            to_numpy(buffer.move_data_to_python()));
            }
        }
  }
}

std::map<uint, uint> simulate_circuit_clifford(py::object const& pycircuit,
                                               const int& shots) {

  auto circuit = Circuit(pycircuit);

  // If measure all at the end, simulate once
  uint actual_shots = shots;

  // qbit, cbit
  vector<std::pair<uint, uint>> measures = circuit.measure_vec();
  std::map<uint, bool> cbit_measured;
  for (auto& pair : measures) {
    cbit_measured[pair.second] = true;
  }

  // Store outcome's count
  std::map<uint, uint> outcount;

  circuit_simulator<_word_size> cs(circuit.qubit_num());

  for (uint i = 0; i < actual_shots; i++) {

    simulate(circuit, cs);
    uint outcome = 0;

    if (!circuit.final_measure()) {
      // qubit, cbit, measure result
      auto measure_results = cs.current_measurement_record();

      // make sure the order is the same with other simulators
      std::sort(
          measure_results.begin(), measure_results.end(),
          [](auto& a, auto& b) { return std::get<1>(a) < std::get<1>(b); });

      for (auto& measure_result : measure_results) {
        outcome *= 2;
        outcome += std::get<2>(measure_result);
      }

    } else if (circuit.final_measure() && !measures.empty()) {
      for (auto& measure : measures) {
        cs.do_circuit_instruction(
            {"measure", std::vector<size_t>{measure.first},
             std::vector<double>{static_cast<double>(measure.second)}});
      }

      // qubit, cbit, measure result
      auto measure_results = cs.current_measurement_record();

      // make sure the order is the same with other simulators
      std::sort(
          measure_results.begin(), measure_results.end(),
          [](auto& a, auto& b) { return std::get<1>(a) < std::get<1>(b); });

      for (auto& measure_result : measure_results) {
        outcome *= 2;
        outcome += std::get<2>(measure_result);
      }
    }

    if (measures.empty()) {
      continue;
    }

    if (outcount.find(outcome) != outcount.end())
      outcount[outcome]++;
    else
      outcount[outcome] = 1;

    cs.reset_tableau();
    cs.sim_record.clear();
  }

  return outcount;
}

#ifdef _USE_GPU
py::object simulate_circuit_gpu(py::object const& pycircuit,
                                py::array_t<complex<double>>& np_inputstate) {
  auto circuit = Circuit(pycircuit);
  py::buffer_info buf = np_inputstate.request();
  auto* data_ptr = reinterpret_cast<std::complex<double>*>(buf.ptr);
  size_t data_size = buf.size;

  if (data_size == 0) {
    StateVector<double> state;
    simulate_gpu(circuit, state);
    return to_numpy(state.move_data_to_python());
  } else {
    StateVector<double> state(data_ptr, buf.size);
    simulate_gpu(circuit, state);
    state.move_data_to_python();
    return np_inputstate;
  }
}
#endif

#ifdef _USE_CUQUANTUM
py::object
simulate_circuit_custate(py::object const& pycircuit,
                         py::array_t<complex<double>>& np_inputstate) {
  auto circuit = Circuit(pycircuit);
  py::buffer_info buf = np_inputstate.request();
  auto* data_ptr = reinterpret_cast<std::complex<double>*>(buf.ptr);
  size_t data_size = buf.size;

  if (data_size == 0) {
    StateVector<double> state;
    simulate_custate(circuit, state);
    return to_numpy(state.move_data_to_python());
  } else {
    StateVector<double> state(data_ptr, buf.size);
    simulate_custate(circuit, state);
    state.move_data_to_python();
    return np_inputstate;
  }
}
#endif

py::object expect_statevec(py::array_t<complex<double>> const&np_inputstate, py::list const paulis)
{
    py::buffer_info buf = np_inputstate.request();
    auto* data_ptr = reinterpret_cast<std::complex<double>*>(buf.ptr);
    size_t data_size = buf.size;
    StateVector<double> state(data_ptr, buf.size);
    py::list pyres;
    for (auto pauli_h : paulis){
         py::object pypauli = py::reinterpret_borrow<py::object>(pauli_h);
         std::vector<pos_t> posv = pypauli.attr("pos").cast<std::vector<pos_t>>();
         string paulistr = pypauli.attr("paulistr").cast<string>();
        double expec = state.expect_pauli(paulistr, posv);
        pyres.attr("append")(expec);
    }
    state.move_data_to_python();
    return pyres;
}

PYBIND11_MODULE(qfvm, m) {
  m.doc() = "Qfvm simulator";
  m.def("simulate_circuit", &simulate_circuit, "Simulate with circuit",
        py::arg("circuit"),
        py::arg("inputstate") = py::array_t<complex<double>>(0),
        py::arg("shots"));
  m.def("simulate_circuit_clifford", &simulate_circuit_clifford,
        "Simulate with circuit using clifford", py::arg("circuit"),
        py::arg("shots"));

  m.def("expect_statevec", &expect_statevec, "Calculate paulis expectation", py::arg("inputstate"), py::arg("paulis"));

  m.def("applyop_statevec", &applyop_statevec, "Apply single operator to state", py::arg("operation"), py::arg("inputstate"));

  m.def("sampling_statevec", &sampling_statevec, "sampling state", py::arg("measures"), py::arg("inputstate"), py::arg("shots"));


#ifdef _USE_GPU
  m.def("simulate_circuit_gpu", &simulate_circuit_gpu, "Simulate with circuit",
        py::arg("circuit"),
        py::arg("inputstate") = py::array_t<complex<double>>(0));
#endif

#ifdef _USE_CUQUANTUM
  m.def("simulate_circuit_custate", &simulate_circuit_custate,
        "Simulate with circuit", py::arg("circuit"),
        py::arg("inputstate") = py::array_t<complex<double>>(0));
#endif
}
