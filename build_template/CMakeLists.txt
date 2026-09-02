# AscendKernelBench fixed CMake build project.
#
# Contract (README 3.1/3.3): every task's LLM-generated operator lives in a
# single self-contained `custom_op.asc`; the built Python extension module is
# always named `custom_op`. The build system is task-agnostic: build.py copies
# this file into the sample work directory and injects the target architecture
# via -DCMAKE_ASC_ARCHITECTURES=<arch> from the hardware profile.
#
# Requires: CANN >= 9.1.0 environment (source <cann>/set_env.sh), torch,
# torch_npu, pybind11 importable by the Python3 interpreter found by CMake.

cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED CMAKE_ASC_ARCHITECTURES OR "${CMAKE_ASC_ARCHITECTURES}" STREQUAL "")
    message(FATAL_ERROR "CMAKE_ASC_ARCHITECTURES is required (e.g. dav-2201, dav-3510)")
endif()
set(CMAKE_ASC_ARCHITECTURES "${CMAKE_ASC_ARCHITECTURES}" CACHE STRING "NPU architecture" FORCE)

find_package(ASC REQUIRED)
project(custom_op_build LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Derive a GCC toolchain root for the ASC/bisheng front-end: when .asc sources
# include torch/extension.h, the LLVM-based compiler needs an explicit
# --gcc-toolchain so C++ standard library headers resolve consistently.
execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libgcc.a
    OUTPUT_VARIABLE LIBGCC_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE GCC_QUERY_RESULT
)
if(NOT GCC_QUERY_RESULT EQUAL 0 OR NOT EXISTS "${LIBGCC_PATH}")
    message(FATAL_ERROR "Failed to locate libgcc.a using ${CMAKE_CXX_COMPILER}")
endif()
get_filename_component(LIBGCC_PATH "${LIBGCC_PATH}" REALPATH)
string(REGEX REPLACE "/lib/gcc/.*$" "" GCC_TOOLCHAIN_ROOT "${LIBGCC_PATH}")
if("${GCC_TOOLCHAIN_ROOT}" STREQUAL "${LIBGCC_PATH}")
    message(FATAL_ERROR "Failed to derive GCC toolchain root from ${LIBGCC_PATH}")
endif()
message(STATUS "GCC toolchain root: ${GCC_TOOLCHAIN_ROOT}")

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E env ASCEND_SLOG_PRINT_TO_STDOUT=0
            ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
)
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})

execute_process(
    COMMAND ${CMAKE_COMMAND} -E env ASCEND_SLOG_PRINT_TO_STDOUT=0
            ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_NPU_PATH
)
set(TORCH_NPU_INCLUDE_DIRS ${TORCH_NPU_PATH}/include)
set(TORCH_NPU_LIBRARIES ${TORCH_NPU_PATH}/lib)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E env ASCEND_SLOG_PRINT_TO_STDOUT=0
            ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYBIND11_CMAKE_PREFIX_PATH
)
find_package(pybind11 REQUIRED HINTS ${PYBIND11_CMAKE_PREFIX_PATH})

# Module name MUST stay `custom_op` and match PYBIND11_MODULE in custom_op.asc.
pybind11_add_module(custom_op SHARED custom_op.asc)

target_include_directories(custom_op PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIRS}
)

target_link_libraries(custom_op PRIVATE
    ${TORCH_LIBRARIES}
    torch_npu
)

target_link_directories(custom_op PRIVATE
    ${TORCH_NPU_LIBRARIES}
)

target_link_options(custom_op PRIVATE
    --gcc-toolchain=${GCC_TOOLCHAIN_ROOT}
)

target_compile_options(custom_op PRIVATE
    ${TORCH_CXX_FLAGS}
    $<$<COMPILE_LANGUAGE:ASC>:--gcc-toolchain=${GCC_TOOLCHAIN_ROOT}>
)
