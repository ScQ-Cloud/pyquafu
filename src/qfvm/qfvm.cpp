#include "simulator.hpp"
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <random>
#ifdef _USE_GPU
#include <cuda_simulator.cuh>
#endif

#ifdef _USE_CUQUANTUM
#include <custate_simu.cuh>
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
  StateVector<double> global_state;
  vector<std::pair<uint, uint>> measures = circuit.measure_vec();
  std::map<uint, bool> cbit_measured;
  for (auto& pair : measures) {
    cbit_measured[pair.second] = true;
  }
  // Store outcome's count
  std::map<uint, uint> outcount;
  for (uint i = 0; i < actual_shots; i++) {
    StateVector<double> state;
    if (data_size == 0) {
      simulate(circuit, state);
    } else {
      // deepcopy state
      vector<std::complex<double>> data_copy(data_ptr, data_ptr + data_size);
      state =
          std::move(StateVector<double>(data_copy.data(), data_copy.size()));
      simulate(circuit, state);
    }
    if (!circuit.final_measure()) {
      // store reg
      vector<uint> tmpcreg = state.creg();
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
    }
    if (circuit.final_measure() || i == actual_shots - 1)
      global_state = std::move(state);
  }
  // sample outcome if final_measure is true
  if (circuit.final_measure() && !measures.empty()) {
    vector<uint> tmpcount(global_state.size(), 0);
    vector<double> probs = global_state.probabilities();
    std::random_device rd;
    std::mt19937 global_rng(rd());
    for (uint i = 0; i < shots; i++) {
      uint outcome = std::discrete_distribution<uint>(probs.begin(),
                                                      probs.end())(global_rng);
      tmpcount[outcome]++;
    }
    // map to reg
    for (uint i = 0; i < global_state.size(); i++) {
      if (tmpcount[i] == 0)
        continue;
      vector<uint> tmpcreg(global_state.cbit_num(), 0);
      vector<uint> tmpout = int2vec(i, 2);
      if (tmpout.size() < global_state.num())
        tmpout.resize(global_state.num());
      for (auto& pair : measures) {
        tmpcreg[pair.second] = tmpout[pair.first];
      }
      uint outcome = 0;
      for (uint j = 0; j < tmpcreg.size(); j++) {
        if (cbit_measured.find(j) == cbit_measured.end())
          continue;
        outcome *= 2;
        outcome += tmpcreg[j];
      }
      if (outcount.find(outcome) != outcount.end())
        outcount[outcome] += tmpcount[i];
      else
        outcount[outcome] = tmpcount[i];
    }
  }
  // return
  if (data_size == 0)
    return std::make_pair(outcount,
                          to_numpy(global_state.move_data_to_python()));
  else
    return std::make_pair(outcount, np_inputstate);
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

PYBIND11_MODULE(qfvm, m) {
  m.doc() = "Qfvm simulator";
  m.def("simulate_circuit", &simulate_circuit, "Simulate with circuit",
        py::arg("circuit"),
        py::arg("inputstate") = py::array_t<complex<double>>(0),
        py::arg("shots"));

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
