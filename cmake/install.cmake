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

include(CMakePackageConfigHelpers)

set(_namespace quafu::)

get_property(quafu_install_targets GLOBAL PROPERTY quafu_install_targets)
list(REMOVE_DUPLICATES quafu_install_targets)

# ==============================================================================

set(QUAFU_INSTALL_IN_BUILD_DIR TRUE)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/quafuConfig.cmake.in ${PROJECT_BINARY_DIR}/quafuConfig.cmake
  INSTALL_DESTINATION ${PROJECT_BINARY_DIR}
  INSTALL_PREFIX ${PROJECT_BINARY_DIR})

# --------------------------------------

set(QUAFU_INSTALL_IN_BUILD_DIR FALSE)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/quafuConfig.cmake.in ${PROJECT_BINARY_DIR}/config_for_install/quafuConfig.cmake
  INSTALL_DESTINATION ${QUAFU_INSTALL_CMAKEDIR})

write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/quafuConfigVersion.cmake
  COMPATIBILITY SameMajorVersion
  VERSION ${QUAFU_VERSION})

install(FILES ${PROJECT_BINARY_DIR}/config_for_install/quafuConfig.cmake
              ${PROJECT_BINARY_DIR}/quafuConfigVersion.cmake DESTINATION ${QUAFU_INSTALL_CMAKEDIR})

# ==============================================================================

install(DIRECTORY ${PROJECT_SOURCE_DIR}/cmake/Modules ${PROJECT_SOURCE_DIR}/cmake/commands
        DESTINATION ${QUAFU_INSTALL_CMAKEDIR})
install(FILES "${CMAKE_CURRENT_LIST_DIR}/packages.cmake" DESTINATION ${QUAFU_INSTALL_CMAKEDIR})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/cmake/NVCXX DESTINATION ${QUAFU_INSTALL_CMAKEDIR}/Modules)

# ------------------------------------------------------------------------------

install(
  TARGETS ${quafu_install_targets}
  EXPORT quafuTargets
  PRIVATE_HEADER DESTINATION ${QUAFU_INSTALL_INCLUDEDIR}
  PUBLIC_HEADER DESTINATION ${QUAFU_INSTALL_INCLUDEDIR}
  ARCHIVE DESTINATION ${QUAFU_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${QUAFU_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${QUAFU_INSTALL_BINDIR})

install(
  EXPORT quafuTargets
  NAMESPACE ${_namespace}
  DESTINATION ${QUAFU_INSTALL_CMAKEDIR})

# NB: if called from setup.py, we do not need to care about installing the Python related targets, as this will be taken
# care of by Python directly.
if(NOT IS_PYTHON_BUILD)
  install(
    EXPORT quafuPythonTargets
    NAMESPACE ${_namespace}
    DESTINATION ${QUAFU_INSTALL_CMAKEDIR})

  export(
    EXPORT quafuPythonTargets
    NAMESPACE ${_namespace}
    FILE quafuPythonTargets.cmake)
endif()

# ==============================================================================

export(
  EXPORT quafuTargets
  NAMESPACE ${_namespace}
  FILE quafuTargets.cmake)
export(PACKAGE quafu)
