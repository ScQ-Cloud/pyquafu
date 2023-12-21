
#pragma once
#include <custatevec.h>
#include <helper_custatevec.hpp>

void apply_gate_custate(cuDoubleComplex* psi_d, QuantumOperator& op, int n) {

  // get information form op
  auto pos = op.positions();
  const int nTargets = op.targe_num();
  const int nControls = op.control_num();
  const int adjoint = 0;

  vector<int> targets{pos.begin() + nControls, pos.end()};
  vector<int> controls{pos.begin(), pos.begin() + nControls};

  auto mat_temp = op.mat();
  cuDoubleComplex* mat = reinterpret_cast<cuDoubleComplex*>(mat_temp.data());

  // custatevec handle initialization
  custatevecHandle_t handle;
  custatevecCreate(&handle);
  void* extraWorkspace = nullptr;
  size_t extraWorkspaceSizeInBytes = 0;

  // check the size of external workspace
  custatevecApplyMatrixGetWorkspaceSize(
      handle, CUDA_C_64F, n, mat, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
      adjoint, nTargets, nControls, CUSTATEVEC_COMPUTE_64F,
      &extraWorkspaceSizeInBytes);

  // allocate external workspace if necessary
  if (extraWorkspaceSizeInBytes > 0)
    cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes);

  custatevecApplyMatrix(handle, psi_d, CUDA_C_64F, n, mat, CUDA_C_64F,
                        CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, targets.data(),
                        nTargets, controls.data(), nullptr, nControls,
                        CUSTATEVEC_COMPUTE_64F, extraWorkspace,
                        extraWorkspaceSizeInBytes);

  // destroy handle
  custatevecDestroy(handle);
}
