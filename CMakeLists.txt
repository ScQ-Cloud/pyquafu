# To use this file to build from source, make sure that external packages openmp, eigen and pybind11 have been installed with cmake by default.   

cmake_minimum_required(VERSION 3.14...3.22)

project(qfvm LANGUAGES CXX C)

set (CMAKE_BUILD_TYPE Release)

if(SKBUILD)

execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
endif()

set ( PRJ_COMPILE_OPTIONS )
set ( PRJ_LIBRARIES )
set ( PRJ_INCLUDE_DIRS )

# openmp
if (${CMAKE_HOST_SYSTEM_NAME} MATCHES Darwin)
    execute_process(COMMAND brew install libomp)
	list ( APPEND PRJ_LIBRARIES /usr/local/opt/libomp/lib )
	list ( APPEND PRJ_INCLUDE_DIRS /usr/local/opt/libomp/include )
	list ( APPEND PRJ_COMPILE_OPTIONS -fopenmp=libomp )
else()
	find_package ( OpenMP )
	if ( OpenMP_FOUND )
		list ( APPEND PRJ_COMPILE_DEF ENABLE_OPENMP )
		list ( APPEND PRJ_LIBRARIES ${OpenMP_CXX_LIBRARIES} )
		list ( APPEND PRJ_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS} )
	endif()
endif()

#install eigen
set(EIGEN3_ROOT  ${CMAKE_SOURCE_DIR}/thirdparty/Eigen3)
set(EIGEN3_INCLUDE_DIR ${EIGEN3_ROOT}/include/eigen3)
include(ExternalProject)
ExternalProject_Add(Eigen3 
		PREFIX             ${EIGEN3_ROOT}
		GIT_REPOSITORY     https://gitlab.com/libeigen/eigen.git
		GIT_TAG            3.3.9
		# CONFIGURE_COMMAND  cd ${EIGEN3_ROOT}/src/Eigen3 && cmake -B build -DCMAKE_INSTALL_PREFIX=${EIGEN3_ROOT}
		# BUILD_COMMAND ""
		# INSTALL_COMMAND    cd ${EIGEN3_ROOT}/src/Eigen3 && cmake --build build --target install

		CONFIGURE_COMMAND ""
		BUILD_COMMAND ""
		INSTALL_COMMAND
      ${CMAKE_COMMAND} -E copy_directory ${EIGEN3_ROOT}/src/Eigen3/Eigen ${EIGEN3_INCLUDE_DIR}/Eigen
      && ${CMAKE_COMMAND} -E copy_directory ${EIGEN3_ROOT}/src/Eigen3/unsupported ${EIGEN3_INCLUDE_DIR}/unsupported
)
list (APPEND PRJ_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

#install pybind11
set(PYBIND11_ROOT ${CMAKE_SOURCE_DIR}/thirdparty/pybind11)
set(PYBIND11_INCLUDE_DIR ${PYBIND11_ROOT}/include)
ExternalProject_Add(pybind11 
		PREFIX             ${PYBIND11_ROOT}
		GIT_REPOSITORY     https://github.com/pybind/pybind11
		# CONFIGURE_COMMAND  cd ${PYBIND11_ROOT}/src/pybind11 && cmake -B build -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=${PYBIND11_ROOT}
		# BUILD_COMMAND ""
		# INSTALL_COMMAND    cd ${PYBIND11_ROOT}/src/pybind11 && cmake --build build --target install

		CONFIGURE_COMMAND ""
		BUILD_COMMAND ""
		INSTALL_COMMAND
		${CMAKE_COMMAND} -E copy_directory ${PYBIND11_ROOT}/src/pybind11/include/pybind11 ${PYBIND11_INCLUDE_DIR}/pybind11
)

find_package(pybind11 CONFIG)
list (APPEND PRJ_INCLUDE_DIRS ${PYBIND11_INCLUDE_DIR})

#SIMD
if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "AMD64" OR CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "amd64")
	if(MSVC)
		list ( APPEND PRJ_COMPILE_OPTIONS /fp:fast /arch:AVX2)
		add_compile_definitions(USE_SIMD)
	else()
		if (NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
			list ( APPEND PRJ_COMPILE_OPTIONS "-mfma;-mavx2")
			list ( APPEND PRJ_COMPILE_OPTIONS -ffast-math)
			add_compile_definitions(USE_SIMD)
		endif()
	endif()
endif()

pybind11_add_module(${PROJECT_NAME} MODULE src/${PROJECT_NAME}/${PROJECT_NAME}.cpp)
target_compile_options(${PROJECT_NAME} PUBLIC ${PRJ_COMPILE_OPTIONS})
target_include_directories(${PROJECT_NAME} PUBLIC ${PRJ_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${PRJ_LIBRARIES})
set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ${PYTHON_MODULE_EXTENSION})
target_compile_definitions(${PROJECT_NAME} PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS ${PROJECT_NAME} DESTINATION .)
