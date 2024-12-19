# ==============================================================================
#
# Copyright 2022 <Huawei Technologies Co., Ltd>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

# lint_cmake: -whitespace/indent

pybind11_add_module(_quafu_vector MODULE ${CMAKE_CURRENT_SOURCE_DIR}/lib/_quafu_vector.cpp
                    OUTPUT_HINT "${QUAFU_PYTHON_PACKAGE_NAME}")

target_include_directories(_quafu_vector PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)
force_at_least_cxx17_workaround(_quafu_vector)
target_link_libraries(_quafu_vector PUBLIC quafu_python_core quafusim_vector_cpu quafusim_stabilizer)

# ------------------------------------------------------------------------------

if(ENABLE_CUDA)
  pybind11_add_module(_quafu_vector_gpu MODULE ${CMAKE_CURRENT_SOURCE_DIR}/lib/_quafu_vector_gpu.cu
                      OUTPUT_HINT "${QUAFU_PYTHON_PACKAGE_NAME}")

  target_include_directories(_quafu_vector_gpu PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)
  force_at_least_cxx17_workaround(_quafu_vector_gpu)
  target_link_libraries(_quafu_vector_gpu PUBLIC quafu_python_core quafusim_vector_gpu)
  set_target_properties(_quafu_vector_gpu PROPERTIES CUDA_RUNTIME_LIBRARY $<IF:$<BOOL:${CUDA_STATIC}>,Static,Shared>)
endif()

# ------------------------------------------------------------------------------

# lint_cmake: -whitespace/indent

pybind11_add_module(_quafu_matrix MODULE ${CMAKE_CURRENT_SOURCE_DIR}/lib/_quafu_matrix.cpp
                    OUTPUT_HINT "${QUAFU_PYTHON_PACKAGE_NAME}")

target_include_directories(_quafu_matrix PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)
force_at_least_cxx17_workaround(_quafu_matrix)
target_link_libraries(_quafu_matrix PUBLIC quafu_python_core quafusim_densitymatrix_cpu)

# ------------------------------------------------------------------------------

add_subdirectory(include)
