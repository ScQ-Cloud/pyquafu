# To use this file to build from source, make sure that external packages openmp, eigen and pybind11 have been installed with cmake by default.

cmake_minimum_required(VERSION 3.14...3.22)

project(qfvm LANGUAGES CXX C)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Build type not set - defaulting to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 70;75;80;90)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
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
		GIT_TAG            3.4

		CONFIGURE_COMMAND ""
		BUILD_COMMAND ""
		INSTALL_COMMAND
      ${CMAKE_COMMAND} -E copy_directory ${EIGEN3_ROOT}/src/Eigen3/Eigen ${EIGEN3_INCLUDE_DIR}/Eigen
      && ${CMAKE_COMMAND} -E copy_directory ${EIGEN3_ROOT}/src/Eigen3/unsupported ${EIGEN3_INCLUDE_DIR}/unsupported
)
list (APPEND PRJ_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

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

list (APPEND PRJ_INCLUDE_DIRS src/qfvm src/qfvm_clifford)
pybind11_add_module(${PROJECT_NAME} MODULE src/${PROJECT_NAME}/${PROJECT_NAME}.cpp)
add_dependencies(${PROJECT_NAME} Eigen3) #must add dependence for ninja
target_compile_options(${PROJECT_NAME} PUBLIC ${PRJ_COMPILE_OPTIONS})
target_include_directories(${PROJECT_NAME} PUBLIC ${PRJ_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${PRJ_LIBRARIES})
set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ${PYTHON_MODULE_EXTENSION})
target_compile_definitions(${PROJECT_NAME} PRIVATE VERSION_INFO=${PROJECT_VERSION})

#GPU version
if (USE_GPU)
	add_compile_definitions(_USE_GPU)
	enable_language(CUDA)
	set_source_files_properties(src/${PROJECT_NAME}/${PROJECT_NAME}.cpp PROPERTIES LANGUAGE CUDA)
	target_link_libraries(${PROJECT_NAME} PUBLIC cudart)
	target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda> )
	target_include_directories(${PROJECT_NAME} PUBLIC src/qfvm_gpu)
	target_include_directories(${PROJECT_NAME} PUBLIC src/qfvm_gpu/cuda_utils)
	target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
	message("cuda_include" ${CUDA_INCLUDE_DIRS})
	if (USE_CUQUANTUM)
		add_compile_definitions(_USE_CUQUANTUM)
		function(set_with_fallback VARIABLE FALLBACK)
			if (NOT DEFINED ${VARIABLE} OR ${VARIABLE} STREQUAL "")
			set(${VARIABLE} $ENV{${VARIABLE}} CACHE INTERNAL ${VARIABLE})
			if (${VARIABLE} STREQUAL "")
				if (NOT ${FALLBACK} STREQUAL "")
				set(${VARIABLE} $ENV{${FALLBACK}} CACHE INTERNAL ${VARIABLE})
				endif ()
			endif ()
			endif ()
		endfunction()

		set_with_fallback(CUSTATEVEC_ROOT CUQUANTUM_ROOT)

		if (CUSTATEVEC_ROOT STREQUAL "")
			message(FATAL_ERROR "Please set the environment variables CUSTATEVEC_ROOT or CUQUANTUM_ROOT to the path of the cuQuantum installation.")
		endif ()

		message(STATUS "Using CUSTATEVEC_ROOT = ${CUSTATEVEC_ROOT}")

		set(CMAKE_CUDA_FLAGS_ARCH_SM70 "-gencode arch=compute_70,code=sm_70")
		set(CMAKE_CUDA_FLAGS_ARCH_SM75 "-gencode arch=compute_75,code=sm_75")
		set(CMAKE_CUDA_FLAGS_ARCH_SM80 "-gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80")
		set(CMAKE_CUDA_FLAGS_ARCH_SM90 "-gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90")
		set(CMAKE_CUDA_FLAGS_ARCH "${CMAKE_CUDA_FLAGS_ARCH_SM70} ${CMAKE_CUDA_FLAGS_ARCH_SM75} ${CMAKE_CUDA_FLAGS_ARCH_SM80} ${CMAKE_CUDA_FLAGS_ARCH_SM90}")
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_ARCH}")

		target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS} ${CUSTATEVEC_ROOT}/include)
		target_link_directories(${PROJECT_NAME} PUBLIC ${CUSTATEVEC_ROOT}/lib ${CUSTATEVEC_ROOT}/lib64)
		target_link_libraries(${PROJECT_NAME} PUBLIC -lcustatevec_static -lcublas )
	endif()
endif()

install(TARGETS ${PROJECT_NAME} DESTINATION .)
