
#pragma once
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <statevector.hpp>

class CudaStateVector {
protected:
  uint num_;
  size_t size_;
  cuDoubleComplex* data_;

public:
  // construct function
  CudaStateVector() { checkCudaErrors(cudaMalloc(&data_, 0)); }
  CudaStateVector(CudaStateVector const& other);

  explicit CudaStateVector(StateVector<double>& sv);
  ~CudaStateVector() { checkCudaErrors(cudaFree(data_)); }

  CudaStateVector& operator=(CudaStateVector const& other) {
    num_ = other.num();
    size_ = other.size();
    checkCudaErrors(cudaMalloc(&data_, size_ * sizeof(cuDoubleComplex)));
    checkCudaErrors(cudaMemcpy(data_, other.data(),
                               size_ * sizeof(cuDoubleComplex),
                               cudaMemcpyDeviceToDevice));
    return *this;
  }

  cuDoubleComplex* data() const { return data_; }
  size_t size() const { return size_; }
  uint num() const { return num_; }
};

CudaStateVector::CudaStateVector(StateVector<double>& sv)
    : num_(sv.num()), size_(sv.size()) {
  cuDoubleComplex* psi_h = reinterpret_cast<cuDoubleComplex*>(sv.data());
  checkCudaErrors(cudaMalloc(&data_, size_ * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMemcpy(data_, psi_h, size_ * sizeof(cuDoubleComplex),
                             cudaMemcpyHostToDevice));
}

CudaStateVector::CudaStateVector(CudaStateVector const& other)
    : num_(other.num()), size_(other.size()) {
  checkCudaErrors(cudaMalloc(&data_, size_ * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMemcpy(data_, other.data(),
                             size_ * sizeof(cuDoubleComplex),
                             cudaMemcpyDeviceToDevice));
}
