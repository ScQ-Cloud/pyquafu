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

# lint_cmake: -whitespace/indent,-whitespace/extra

add_library(quafubackend_python_interface INTERFACE)
target_include_directories(quafubackend_python_interface INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
                                                                $<INSTALL_INTERFACE:${QUAFU_INSTALL_INCLUDEDIR}>)
append_to_property(quafu_install_targets GLOBAL quafubackend_python_interface)

# ------------------------------------------------------------------------------

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/python DESTINATION ${QUAFU_INSTALL_INCLUDEDIR})

# ==============================================================================

set(_lto_workaround FALSE)
if((CMAKE_COMPILER_IS_GNUCXX
    AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0.0
    AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7.3.0)
   OR (MSVC
       AND COMPILER_IS_CLANG_CL
       AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 15.0.0))
  # NB: enable workaround for GCC 7.X or Clang-Cl < 15.0.0
  set(_lto_workaround TRUE)
endif()

if(_lto_workaround)
  # NB: workaround some internal compiler error linked to LTO
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
endif()

pybind11_add_module(
  quafubackend MODULE
  OUTPUT_HINT
    "${QUAFU_PYTHON_PACKAGE_NAME}" ${CMAKE_CURRENT_LIST_DIR}/lib/binding.cc ${CMAKE_CURRENT_LIST_DIR}/lib/logging.cpp
                                ${CMAKE_CURRENT_LIST_DIR}/lib/build_env.cpp)

if(_lto_workaround)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

target_include_directories(quafubackend PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)

force_at_least_cxx17_workaround(quafubackend)
target_link_libraries(quafubackend PUBLIC ${QUAFU_OPENMP_TARGET} quafu_base quafu_python_core quafubackend_python_interface
                                       quafu_python_device)

# ------------------------------------------------------------------------------

get_target_property(_quafubackend_output_dir quafubackend LIBRARY_OUTPUT_DIRECTORY)
if(NOT _quafubackend_output_dir)
  set(_quafubackend_output_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
endif()
