#pragma once
#include <algorithm>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <types.hpp>
#include <util.h>

struct targeIndex {
  size_t ind0;
  size_t ind1;
};

__constant__ uint posv_d[50];
__constant__ uint posv_sorted_d[50];
__constant__ cuDoubleComplex
    mat_d_const[32 * 32]; // If target qubit < 5, use const memory;
__constant__ uint mat_mask_d_const[32];

//-------------Single-target gate-------------------------------
template <class Func>
__global__ void apply_one_targe_gate_kernel(cuDoubleComplex* psi_d,
                                            Func get_index, int rsize) {
  cuDoubleComplex mat00 = mat_d_const[0];
  cuDoubleComplex mat01 = mat_d_const[1];
  cuDoubleComplex mat10 = mat_d_const[2];
  cuDoubleComplex mat11 = mat_d_const[3];

  unsigned int gridSize = blockDim.x * gridDim.x;
  for (int j = blockDim.x * blockIdx.x + threadIdx.x; j < rsize;
       j += gridSize) {
    targeIndex ind = get_index(j);
    cuDoubleComplex temp = psi_d[ind.ind0];
    psi_d[ind.ind0] =
        cuCadd(cuCmul(mat00, psi_d[ind.ind0]), cuCmul(mat01, psi_d[ind.ind1]));
    psi_d[ind.ind1] =
        cuCadd(cuCmul(mat10, temp), cuCmul(mat11, psi_d[ind.ind1]));
  }
}

template <int ctrl_num>
void apply_one_targe_gate_gpu(cuDoubleComplex* psi_d, QuantumOperator& op,
                              size_t size) {
  // copy mat to device
  auto mat_temp = op.mat();
  cuDoubleComplex* mat = reinterpret_cast<cuDoubleComplex*>(mat_temp.data());
  checkCudaErrors(
      cudaMemcpyToSymbol(mat_d_const, mat, 4 * sizeof(cuDoubleComplex)));
  size_t rsize;
  size_t offset;
  size_t targe;
  size_t control;
  size_t setbit;
  size_t poffset;
  if (ctrl_num == 0) {
    targe = op.positions()[0];
    offset = 1ll << targe;
    rsize = size >> 1;
    auto getind_func = [offset, targe] __device__(size_t j) -> targeIndex {
      size_t ind0 = (j & (offset - 1)) | (j >> targe << targe << 1);
      size_t ind1 = ind0 + offset;
      return {ind0, ind1};
    };

    size_t blockdim = rsize <= 1024 ? rsize : 1024;
    size_t griddim = rsize / blockdim;
    apply_one_targe_gate_kernel<<<griddim, blockdim>>>(psi_d, getind_func,
                                                       rsize);
  } else if (ctrl_num == 1) {
    control = op.positions()[0];
    targe = op.positions()[1];
    offset = 1ll << targe;
    setbit = 1ll << control;
    if (control > targe) {
      control--;
    }
    poffset = 1ll << control;
    rsize = size >> 2;
    auto getind_func = [control, targe, poffset, offset,
                        setbit] __device__(size_t j) -> targeIndex {
      size_t ind0 = (j >> control << (control + 1)) | (j & (poffset - 1));
      ind0 = (ind0 >> targe << (targe + 1)) | (ind0 & (offset - 1)) | setbit;
      size_t ind1 = ind0 + offset;
      return {ind0, ind1};
    };

    size_t blockdim = rsize <= 1024 ? rsize : 1024;
    size_t griddim = rsize / blockdim;

    apply_one_targe_gate_kernel<<<griddim, blockdim>>>(psi_d, getind_func,
                                                       rsize);
  } else if (ctrl_num == 2) {
    targe = op.positions().back();
    offset = 1ll << targe;
    uint psize = op.positions().size();
    rsize = size >> psize;

    vector<pos_t> posv_sorted = op.positions();
    std::sort(posv_sorted.begin(), posv_sorted.end());
    // Copy pos to device
    checkCudaErrors(cudaMemcpyToSymbol(posv_d, op.positions().data(),
                                       psize * sizeof(uint)));
    checkCudaErrors(cudaMemcpyToSymbol(posv_sorted_d, posv_sorted.data(),
                                       psize * sizeof(uint)));

    auto getind_func = [offset, psize] __device__(size_t j) -> targeIndex {
      size_t ind0 = j;
      for (pos_t k = 0; k < psize; k++) {
        pos_t _pos = posv_sorted_d[k];
        ind0 = (ind0 & ((1ll << _pos) - 1)) | (ind0 >> _pos << _pos << 1);
      }
      for (pos_t k = 0; k < psize - 1; k++) {
        ind0 |= 1ll << posv_d[k];
      }

      size_t ind1 = ind0 + offset;
      return {ind0, ind1};
    };
    size_t blockdim = rsize <= 1024 ? rsize : 1024;
    size_t griddim = rsize / blockdim;
    apply_one_targe_gate_kernel<<<griddim, blockdim>>>(psi_d, getind_func,
                                                       rsize);
  }
}

template <int targe_num>
__global__ void apply_2to4_targe_gate_kernel(cuDoubleComplex* psi_d,
                                             uint ctrlnum, int psize) {
  constexpr uint matlen = 1 << targe_num;
  uint block_length = blockDim.x;
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  // Insert zeros
  for (size_t k = 0; k < psize; k++) {
    size_t _pos = posv_sorted_d[k];
    i = (i & ((1ll << _pos) - 1)) | (i >> _pos << _pos << 1);
  }
  // Set control
  for (size_t k = 0; k < ctrlnum; k++) {
    i |= 1ll << posv_d[k];
  }

  cuDoubleComplex psi_d_buffer[matlen];
  for (int y = 0; y < matlen; ++y) {
    psi_d_buffer[y] = {0., 0.};
    for (int x = 0; x < matlen; ++x) {
      psi_d_buffer[y] =
          cuCadd(psi_d_buffer[y], cuCmul(psi_d[i | mat_mask_d_const[x]],
                                         mat_d_const[y * matlen + x]));
    }
  }
  for (int y = 0; y < matlen; ++y)
    psi_d[i | mat_mask_d_const[y]] = psi_d_buffer[y];
}

template <int targe_num>
void apply_2to4_targe_gate_gpu_const(cuDoubleComplex* psi_d,
                                     QuantumOperator& op, size_t size) {
  // uint targe_num = op.targe_num();
  uint matlen = 1 << targe_num;
  auto pos = op.positions();
  auto targs = vector<pos_t>(pos.begin() + op.control_num(), pos.end());
  vector<uint> targ_mask(matlen);
  // create target mask
  for (size_t m = 0; m < matlen; m++) {
    for (size_t j = 0; j < targe_num; j++) {
      if ((m >> j) & 1) {
        auto mask_pos = targs[j];
        targ_mask[m] |= 1ll << mask_pos;
      }
    }
  }

  vector<pos_t> posv_sorted = op.positions();
  uint psize = pos.size();
  std::sort(posv_sorted.begin(), posv_sorted.end());
  // Copy pos to device
  checkCudaErrors(cudaMemcpyToSymbol(posv_d, pos.data(), psize * sizeof(uint)));
  checkCudaErrors(cudaMemcpyToSymbol(posv_sorted_d, posv_sorted.data(),
                                     psize * sizeof(uint)));

  // copy mat to const memory
  auto mat_temp = op.mat();
  cuDoubleComplex* mat = reinterpret_cast<cuDoubleComplex*>(mat_temp.data());

  checkCudaErrors(cudaMemcpyToSymbol(
      mat_d_const, mat, matlen * matlen * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMemcpyToSymbol(mat_mask_d_const, targ_mask.data(),
                                     matlen * sizeof(uint)));
  size_t rsize = size >> psize;

  uint max_thread_num = targe_num < 4 ? 1024 : 512;
  size_t blockdim = rsize <= max_thread_num ? rsize : max_thread_num;
  size_t griddim = rsize / blockdim;
  apply_2to4_targe_gate_kernel<targe_num>
      <<<griddim, blockdim>>>(psi_d, op.control_num(), psize);
}

// ------------Large target number gate---------------

__global__ void apply_5_targe_gate_kernel_const(cuDoubleComplex* psi_d,
                                                uint ctrlnum, int psize,
                                                size_t size) {
  uint rsize = size >> psize;
  uint targnum = psize - ctrlnum;
  uint matlen = (1 << targnum);
  uint block_length = blockDim.x;
  size_t b = blockIdx.x; // < rsize
  int idx = threadIdx.x; //
  int idy = threadIdx.y; //
  size_t i = b;
  // Insert zeros
  for (size_t k = 0; k < psize; k++) {
    size_t _pos = posv_sorted_d[k];
    i = (i & ((1ll << _pos) - 1)) | (i >> _pos << _pos << 1);
  }
  // Set control
  for (size_t k = 0; k < ctrlnum; k++) {
    i |= 1ll << posv_d[k];
  }

  __syncthreads();
  cuDoubleComplex v;
  v = cuCmul(psi_d[i | mat_mask_d_const[idx]], mat_d_const[idy * matlen + idx]);
  for (int offset = block_length >> 1; offset > 0; offset >>= 1) {
    v.x += __shfl_down_sync(0xFFFFFFFF, v.x, offset);
    v.y += __shfl_down_sync(0xFFFFFFFF, v.y, offset);
  }
  __syncthreads();
  if (!idx)
    psi_d[i | mat_mask_d_const[idy]] = v;
}

void apply_5_targe_gate_gpu_const(cuDoubleComplex* psi_d, QuantumOperator& op,
                                  size_t size) {
  uint targe_num = op.targe_num();
  uint matlen = 1 << targe_num;
  auto pos = op.positions();
  auto targs = vector<pos_t>(pos.begin() + op.control_num(), pos.end());
  vector<uint> targ_mask(matlen);
  // create target mask
  for (size_t m = 0; m < matlen; m++) {
    for (size_t j = 0; j < targe_num; j++) {
      if ((m >> j) & 1) {
        auto mask_pos = targs[j];
        targ_mask[m] |= 1ll << mask_pos;
      }
    }
  }

  vector<pos_t> posv_sorted = op.positions();
  uint psize = pos.size();
  std::sort(posv_sorted.begin(), posv_sorted.end());
  // Copy pos to device
  checkCudaErrors(cudaMemcpyToSymbol(posv_d, pos.data(), psize * sizeof(uint)));
  checkCudaErrors(cudaMemcpyToSymbol(posv_sorted_d, posv_sorted.data(),
                                     psize * sizeof(uint)));

  // copy mat to const memory
  auto mat_temp = op.mat();
  cuDoubleComplex* mat = reinterpret_cast<cuDoubleComplex*>(mat_temp.data());

  checkCudaErrors(cudaMemcpyToSymbol(
      mat_d_const, mat, matlen * matlen * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMemcpyToSymbol(mat_mask_d_const, targ_mask.data(),
                                     matlen * sizeof(uint)));
  size_t rsize = size >> psize;
  uint thread_num = matlen > 32 ? 32 : matlen;
  dim3 blockdim = dim3(thread_num, thread_num);
  apply_5_targe_gate_kernel_const<<<rsize, blockdim,
                                    thread_num * sizeof(cuDoubleComplex)>>>(
      psi_d, op.control_num(), psize, size);
}

// For target number 6-10
__global__ void apply_multi_targe_gate_kernel_shared(cuDoubleComplex* psi_d,
                                                     uint ctrlnum,
                                                     cuDoubleComplex* mat_d,
                                                     uint* mat_mask_d,
                                                     int psize, size_t size) {

  uint rsize = size >> psize;
  uint targnum = psize - ctrlnum;
  uint matlen = (1 << targnum);
  uint block_length = blockDim.x;
  size_t b = blockIdx.x; // < rsize
  int idx = threadIdx.x; //
  int idy = threadIdx.y; //
  size_t i = b;
  // Insert zeros
  for (size_t k = 0; k < psize; k++) {
    size_t _pos = posv_sorted_d[k];
    i = (i & ((1ll << _pos) - 1)) | (i >> _pos << _pos << 1);
  }
  // Set control
  for (size_t k = 0; k < ctrlnum; k++) {
    i |= 1ll << posv_d[k];
  }

  __syncthreads();
  __shared__ cuDoubleComplex local_sum[1024];
  cuDoubleComplex v;
  for (int y = idy; y < matlen; y += blockDim.y) {
    local_sum[y] = {0, 0};
    for (int x = idx; x < matlen; x += blockDim.x) {
      v = cuCmul(psi_d[i | mat_mask_d[x]], mat_d[y * matlen + x]);
      __syncthreads();
      for (int offset = block_length >> 1; offset > 0; offset >>= 1) {
        v.x += __shfl_down_sync(0xFFFFFFFF, v.x, offset);
        v.y += __shfl_down_sync(0xFFFFFFFF, v.y, offset);
      }
      __syncthreads();
      if (!idx)
        local_sum[y] = cuCadd(local_sum[y], v);
    }
  }

  for (int y = idy; y < matlen; y += blockDim.y) {
    if (!idx)
      psi_d[i | mat_mask_d[y]] = local_sum[y];
  }
}

void apply_multi_targe_gate_gpu_shared(cuDoubleComplex* psi_d,
                                       QuantumOperator& op,
                                       cuDoubleComplex* mat_d, uint* mat_mask_d,
                                       size_t size) {
  uint targe_num = op.targe_num();
  uint matlen = 1 << targe_num;
  auto pos = op.positions();
  uint psize = pos.size();
  auto targs = vector<pos_t>(pos.begin() + op.control_num(), pos.end());
  vector<uint> targ_mask(matlen);
  // create target mask
  for (size_t m = 0; m < matlen; m++) {
    for (size_t j = 0; j < targe_num; j++) {
      if ((m >> j) & 1) {
        auto mask_pos = targs[j];
        targ_mask[m] |= 1ll << mask_pos;
      }
    }
  }

  vector<pos_t> posv_sorted = pos;
  std::sort(posv_sorted.begin(), posv_sorted.end());
  // Copy pos to device
  checkCudaErrors(cudaMemcpyToSymbol(posv_d, pos.data(), psize * sizeof(uint)));
  checkCudaErrors(cudaMemcpyToSymbol(posv_sorted_d, posv_sorted.data(),
                                     psize * sizeof(uint)));

  // copy mat to global memory
  auto mat_temp = op.mat();
  cuDoubleComplex* mat = reinterpret_cast<cuDoubleComplex*>(mat_temp.data());
  checkCudaErrors(cudaMemcpy(mat_d, mat,
                             matlen * matlen * sizeof(cuDoubleComplex),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(mat_mask_d, targ_mask.data(),
                             matlen * sizeof(uint), cudaMemcpyHostToDevice));

  size_t rsize = size >> psize;
  uint thread_num = matlen > 32 ? 32 : matlen;
  dim3 blockdim = dim3(thread_num, thread_num);

  apply_multi_targe_gate_kernel_shared<<<rsize, blockdim>>>(
      psi_d, op.control_num(), mat_d, mat_mask_d, psize, size);
}

// For target number > 10
__global__ void apply_multi_targe_gate_kernel_global(
    cuDoubleComplex* psi_d, cuDoubleComplex* psi_d_copy, uint ctrlnum,
    cuDoubleComplex* mat_d, uint* mat_mask_d, int psize, size_t size) {
  uint rsize = size >> psize;
  uint targnum = psize - ctrlnum;
  uint matlen = (1 << targnum);
  uint block_length = blockDim.x;
  size_t b = blockIdx.x; // < rsize
  int idx = threadIdx.x; //
  int idy = threadIdx.y; //
  size_t i = b;
  // Insert zeros
  for (size_t k = 0; k < psize; k++) {
    size_t _pos = posv_sorted_d[k];
    i = (i & ((1ll << _pos) - 1)) | (i >> _pos << _pos << 1);
  }
  // Set control
  for (size_t k = 0; k < ctrlnum; k++) {
    i |= 1ll << posv_d[k];
  }

  __syncthreads();

  cuDoubleComplex v;
  cuDoubleComplex v_sum;
  for (int y = idy; y < matlen; y += blockDim.y) {
    v_sum = {0, 0};
    for (int x = idx; x < matlen; x += blockDim.x) {
      v = cuCmul(psi_d_copy[i | mat_mask_d[x]], mat_d[y * matlen + x]);
      __syncthreads();
      for (int offset = block_length >> 1; offset > 0; offset >>= 1) {
        v.x += __shfl_down_sync(0xFFFFFFFF, v.x, offset);
        v.y += __shfl_down_sync(0xFFFFFFFF, v.y, offset);
      }
      __syncthreads();
      if (!idx)
        v_sum = cuCadd(v_sum, v);
    }
    if (!idx)
      psi_d[i | mat_mask_d[y]] = v_sum;
  }
}

void apply_multi_targe_gate_gpu_global(cuDoubleComplex* psi_d,
                                       cuDoubleComplex* psi_d_copy,
                                       QuantumOperator& op,
                                       cuDoubleComplex* mat_d, uint* mat_mask_d,
                                       size_t size) {
  uint targe_num = op.targe_num();
  uint matlen = 1 << targe_num;
  auto pos = op.positions();
  uint psize = pos.size();
  auto targs = vector<pos_t>(pos.begin() + op.control_num(), pos.end());
  vector<uint> targ_mask(matlen);
  // create target mask
  for (size_t m = 0; m < matlen; m++) {
    for (size_t j = 0; j < targe_num; j++) {
      if ((m >> j) & 1) {
        auto mask_pos = targs[j];
        targ_mask[m] |= 1ll << mask_pos;
      }
    }
  }

  vector<pos_t> posv_sorted = pos;
  std::sort(posv_sorted.begin(), posv_sorted.end());
  // Copy pos to device
  checkCudaErrors(cudaMemcpyToSymbol(posv_d, pos.data(), psize * sizeof(uint)));
  checkCudaErrors(cudaMemcpyToSymbol(posv_sorted_d, posv_sorted.data(),
                                     psize * sizeof(uint)));

  // copy mat to global memory
  auto mat_temp = op.mat();
  cuDoubleComplex* mat = reinterpret_cast<cuDoubleComplex*>(mat_temp.data());
  checkCudaErrors(cudaMemcpy(mat_d, mat,
                             matlen * matlen * sizeof(cuDoubleComplex),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(mat_mask_d, targ_mask.data(),
                             matlen * sizeof(uint), cudaMemcpyHostToDevice));

  size_t rsize = size >> psize;
  uint thread_num = matlen > 32 ? 32 : matlen;
  dim3 blockdim = dim3(thread_num, thread_num);

  apply_multi_targe_gate_kernel_global<<<rsize, blockdim>>>(
      psi_d, psi_d_copy, op.control_num(), mat_d, mat_mask_d, psize, size);
  checkCudaErrors(cudaMemcpy(psi_d_copy, psi_d, size * sizeof(cuDoubleComplex),
                             cudaMemcpyDeviceToDevice));
}
