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

add_library(quafusim_common STATIC ${CMAKE_CURRENT_LIST_DIR}/utils.cpp ${CMAKE_CURRENT_LIST_DIR}/timer.cpp)
target_link_libraries(quafusim_common PUBLIC quafu_base)
force_at_least_cxx17_workaround(quafusim_common)
append_to_property(quafu_install_targets GLOBAL quafusim_common)
if(MSVC)
  target_compile_options(quafusim_common PUBLIC /wd4661 /openmp:llvm)
endif()

add_library(quafusim_vector_cpu STATIC)
target_link_libraries(quafusim_vector_cpu PUBLIC quafusim_common quafu_math intrin_flag_CXX)
force_at_least_cxx17_workaround(quafusim_vector_cpu)
append_to_property(quafu_install_targets GLOBAL quafusim_vector_cpu)
if(X86_64)
  add_executable(quafurt)
  target_link_libraries(quafurt PUBLIC quafusim_vector_cpu)
  force_at_least_cxx17_workaround(quafurt)
  append_to_property(quafu_install_targets GLOBAL quafurt)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/vector/detail/runtime)
endif()
# ==============================================================================

# add_library( quafusim_vector_cpu STATIC ${CMAKE_CURRENT_LIST_DIR}/vector/detail/cpu_vector_policy.cpp
# ${CMAKE_CURRENT_LIST_DIR}/vector/detail/cpu_vector_policy_zlike.cpp
# ${CMAKE_CURRENT_LIST_DIR}/vector/detail/cpu_vector_policy_xlike.cpp
# ${CMAKE_CURRENT_LIST_DIR}/vector/detail/cpu_vector_policy_single_ops.cpp
# ${CMAKE_CURRENT_LIST_DIR}/vector/detail/cpu_vector_policy_gate_expec.cpp)

# target_link_libraries(quafusim_vector_cpu PUBLIC quafusim_common intrin_flag_CXX)
# force_at_least_cxx17_workaround(quafusim_vector_cpu) append_to_property(quafu_install_targets GLOBAL quafusim_vector_cpu)

# ==============================================================================

if(ENABLE_CUDA)
  add_library(quafusim_vector_gpu STATIC)
  target_link_libraries(quafusim_vector_gpu PUBLIC quafusim_common CUDA_quafu)
  set_target_properties(quafusim_vector_gpu PROPERTIES CUDA_RUNTIME_LIBRARY $<IF:$<BOOL:${CUDA_STATIC}>,Static,Shared>)
  force_at_least_cxx17_workaround(quafusim_vector_gpu)
  append_to_property(quafu_install_targets GLOBAL quafusim_vector_gpu)
endif()

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/vector/detail)

# ==============================================================================

add_library(quafusim_densitymatrix_cpu STATIC)
target_link_libraries(quafusim_densitymatrix_cpu PUBLIC quafusim_common quafu_math intrin_flag_CXX)
force_at_least_cxx17_workaround(quafusim_densitymatrix_cpu)
append_to_property(quafu_install_targets GLOBAL quafusim_densitymatrix_cpu)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/densitymatrix/detail)

# ==============================================================================

add_library(quafusim_stabilizer STATIC)
target_link_libraries(quafusim_stabilizer PUBLIC quafusim_common quafu_math intrin_flag_CXX)
force_at_least_cxx17_workaround(quafusim_stabilizer)
append_to_property(quafu_install_targets GLOBAL quafusim_stabilizer)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/stabilizer)
