#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "simulator.hpp"

namespace py = pybind11;

template <typename T>
py::array_t<T> to_numpy(std::vector<T> &&src) {
  vector<T>* src_ptr = new std::vector<T>(std::move(src));
  auto capsule = py::capsule(src_ptr, [](void* p) { delete reinterpret_cast<std::vector<T>*>(p); });
  return py::array_t<T>(
    src_ptr->size(),  // shape of array
    src_ptr->data(),  // c-style contiguous strides for vector
    capsule           // numpy array references this parent
  );
}


py::object execute(string qasm){
    return to_numpy(simulate(qasm).move_data());
}

py::object simulate_circuit(py::object const&pycircuit, vector<complex<double>> const&inputstate){
  auto circuit = Circuit(pycircuit);
    if (inputstate.size() == 0){
        StateVector<double> state;
        simulate(circuit, state);
        return to_numpy(state.move_data());
    }
    else{
      StateVector<double> state{inputstate};
      simulate(circuit, state);
      return to_numpy(state.move_data());
    }
}


PYBIND11_MODULE(qfvm, m) {
    m.doc() = "Qfvm simulator";
    m.def("execute", &execute, "Simulate with qasm");
    m.def("simulate_circuit", &simulate_circuit, "Simulate with circuit", py::arg("circuit"), py::arg("inputstate")= py::array_t<complex<double>>(0));
}

