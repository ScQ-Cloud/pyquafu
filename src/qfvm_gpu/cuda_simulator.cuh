#pragma once
#include "apply_gate_gpu.cuh"
#include "cuda_statevector.cuh"
#include <circuit.hpp>
#include <statevector.hpp>
#include <ticktock.h>
#include <types.hpp>

void simulate_gpu(Circuit& circuit, CudaStateVector& psi_d) {
  size_t size = psi_d.size();
  // initialize mat
  cuDoubleComplex* mat_d;
  uint* mat_mask_d;
  CudaStateVector psi_d_copy{};
  if (circuit.max_targe_num() > 5) {
    uint max_matlen = 1 << circuit.max_targe_num();
    checkCudaErrors(cudaMalloc(&mat_d, (max_matlen * max_matlen) *
                                           sizeof(cuDoubleComplex)));
    checkCudaErrors(
        cudaMalloc(&mat_mask_d, max_matlen * sizeof(cuDoubleComplex)));
    if (circuit.max_targe_num() > 10) {
      psi_d_copy = psi_d;
    }
  }

  // apply_gate
  for (auto gate : circuit.gates()) {
    uint targnum = gate.targe_num();
    uint ctrlnum = gate.control_num();

    if (targnum == 1) {
      if (ctrlnum == 0) {
        apply_one_targe_gate_gpu<0>(psi_d.data(), gate, size);
      } else if (ctrlnum == 1) {
        apply_one_targe_gate_gpu<1>(psi_d.data(), gate, size);
      } else {
        apply_one_targe_gate_gpu<2>(psi_d.data(), gate, size);
      }
    } else if (targnum > 1) {
      if (targnum == 2) {
        apply_2to4_targe_gate_gpu_const<2>(psi_d.data(), gate, size);
      } else if (targnum == 3) {
        apply_2to4_targe_gate_gpu_const<3>(psi_d.data(), gate, size);
      } else if (targnum == 4) {
        apply_2to4_targe_gate_gpu_const<4>(psi_d.data(), gate, size);
      } else if (targnum == 5) {
        apply_5_targe_gate_gpu_const(psi_d.data(), gate, size);
      } else if (targnum > 5 && targnum <= 10) {
        apply_multi_targe_gate_gpu_shared(psi_d.data(), gate, mat_d, mat_mask_d,
                                          size);
      } else {
        apply_multi_targe_gate_gpu_global(psi_d.data(), psi_d_copy.data(), gate,
                                          mat_d, mat_mask_d, size);
      }
    } else {
      throw "Invalid target number";
    }
  }

  // free source
  if (circuit.max_targe_num() > 5) {
    checkCudaErrors(cudaFree(mat_d));
    checkCudaErrors(cudaFree(mat_mask_d));
  }
}

void simulate_gpu(Circuit& circuit, StateVector<data_t>& state) {
  // initialize psi
  state.set_num(circuit.qubit_num());
  size_t size = state.size();
  CudaStateVector psi_d(state);

  simulate_gpu(circuit, psi_d);
  cudaDeviceSynchronize();

  // copy back
  complex<double>* psi = reinterpret_cast<complex<double>*>(psi_d.data());
  checkCudaErrors(cudaMemcpy(state.data(), psi, size * sizeof(complex<double>),
                             cudaMemcpyDeviceToHost));
  psi = nullptr;
}

StateVector<double> simulate_gpu(Circuit& circuit) {
  StateVector<double> state(circuit.qubit_num());
  simulate_gpu(circuit, state);
  return std::move(state);
}
