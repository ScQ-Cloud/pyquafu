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


PYBIND11_MODULE(qfvm, m) {
    m.doc() = "Qfsim simulator";
    m.def("execute", &execute, "Simulate qasm");
}

