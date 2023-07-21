#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "simulator.hpp"

#ifdef _USE_GPU
#include <cuda_simulator.cuh>
#endif

#ifdef _USE_CUQUANTUM
#include <custate_simu.cuh>
#endif

namespace py = pybind11;

template <typename T>
py::array_t<T> to_numpy(const std::tuple<T*, size_t> &src) {
    auto src_ptr = std::get<0>(src);
    auto src_size = std::get<1>(src);

    auto capsule = py::capsule(src_ptr, [](void* p) {
        delete [] reinterpret_cast<T*>(p);
    });
    return py::array_t<T>(
        src_size,
        src_ptr,
        capsule
    );
}

py::object execute(string qasm){
    return to_numpy(simulate(qasm).move_data_to_python());
}

py::object simulate_circuit(py::object const&pycircuit, py::array_t<complex<double>> &np_inputstate){
    auto circuit = Circuit(pycircuit);
    py::buffer_info buf = np_inputstate.request();
    auto* data_ptr = reinterpret_cast<std::complex<double>*>(buf.ptr);
    size_t data_size = buf.size;

    if (data_size == 0){
        StateVector<double> state;
        simulate(circuit, state);
        return to_numpy(state.move_data_to_python());
    }
    else{
      StateVector<double> state(data_ptr, buf.size);
      simulate(circuit, state);
      state.move_data_to_python();
      return np_inputstate;
    }
}

#ifdef _USE_GPU
py::object simulate_circuit_gpu(py::object const&pycircuit, py::array_t<complex<double>> &np_inputstate){
    auto circuit = Circuit(pycircuit);
    py::buffer_info buf = np_inputstate.request();
    auto* data_ptr = reinterpret_cast<std::complex<double>*>(buf.ptr);
    size_t data_size = buf.size;


    if (data_size == 0){
        StateVector<double> state;
        simulate_gpu(circuit, state);
        return to_numpy(state.move_data_to_python());
    }
    else{
      StateVector<double> state(data_ptr, buf.size);
      simulate_gpu(circuit, state);
      state.move_data_to_python();
      return np_inputstate;
    }
}
#endif

#ifdef _USE_CUQUANTUM
py::object simulate_circuit_custate(py::object const&pycircuit, py::array_t<complex<double>> &np_inputstate){
    auto circuit = Circuit(pycircuit);
    py::buffer_info buf = np_inputstate.request();
    auto* data_ptr = reinterpret_cast<std::complex<double>*>(buf.ptr);
    size_t data_size = buf.size;


    if (data_size == 0){
        StateVector<double> state;
        simulate_custate(circuit, state);
        return to_numpy(state.move_data_to_python());
    }
    else{
      StateVector<double> state(data_ptr, buf.size);
      simulate_custate(circuit, state);
      state.move_data_to_python();
      return np_inputstate;
    }
}
#endif



PYBIND11_MODULE(qfvm, m) {
    m.doc() = "Qfvm simulator";
    m.def("execute", &execute, "Simulate with qasm");
    m.def("simulate_circuit", &simulate_circuit, "Simulate with circuit", py::arg("circuit"), py::arg("inputstate")= py::array_t<complex<double>>(0));

    #ifdef _USE_GPU
     m.def("simulate_circuit_gpu", &simulate_circuit_gpu, "Simulate with circuit", py::arg("circuit"), py::arg("inputstate")= py::array_t<complex<double>>(0));
    #endif

    #ifdef _USE_CUQUANTUM
    m.def("simulate_circuit_custate", &simulate_circuit_custate, "Simulate with circuit", py::arg("circuit"), py::arg("inputstate")= py::array_t<complex<double>>(0));
    #endif
}

