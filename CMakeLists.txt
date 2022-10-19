cmake_minimum_required(VERSION 3.14...3.22)

#cross compile
# include(cmake/toolchain_win_64.cmake)
project(qfvm VERSION "0.0.1" LANGUAGES CXX C)

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

# openmp
find_package ( OpenMP )
if ( OpenMP_FOUND )
	list ( APPEND PRJ_COMPILE_DEF ENABLE_OPENMP )
	list ( APPEND PRJ_LIBRARIES ${OpenMP_CXX_LIBRARIES} )
	list ( APPEND PRJ_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS} )
endif ()


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


find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(${PROJECT_NAME} MODULE src/${PROJECT_NAME}/${PROJECT_NAME}.cpp)
target_compile_options(${PROJECT_NAME} PUBLIC ${PRJ_COMPILE_OPTIONS})
target_include_directories(${PROJECT_NAME} PUBLIC ${PRJ_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${PRJ_LIBRARIES})
set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ${PYTHON_MODULE_EXTENSION})
target_compile_definitions(${PROJECT_NAME} PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS ${PROJECT_NAME} DESTINATION .)