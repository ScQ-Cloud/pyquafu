#pragma once
#include "apply_gate_custate.cuh"
#include "cuda_statevector.cuh"
#include <circuit.hpp>
#include <statevector.hpp>
#include <types.hpp>

void simulate_custate(Circuit& circuit, CudaStateVector& psi_d) {
  size_t size = psi_d.size();
  int n = psi_d.num();
  for (auto gate : circuit.gates()) {
    apply_gate_custate(psi_d.data(), gate, n);
  }
}

void simulate_custate(Circuit& circuit, StateVector<data_t>& state) {
  // initialize psi
  state.set_num(circuit.qubit_num());
  size_t size = state.size();
  CudaStateVector psi_d(state);

  simulate_custate(circuit, psi_d);
  cudaDeviceSynchronize();

  // copy back
  complex<double>* psi = reinterpret_cast<complex<double>*>(psi_d.data());
  checkCudaErrors(cudaMemcpy(state.data(), psi, size * sizeof(complex<double>),
                             cudaMemcpyDeviceToHost));
  psi = nullptr;
}

StateVector<double> simulate_custate(Circuit& circuit) {
  StateVector<double> state(circuit.qubit_num());
  simulate_custate(circuit, state);
  return std::move(state);
}
