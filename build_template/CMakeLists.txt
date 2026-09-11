# AscendKernelBench ACLNN operator project (process-local shared library).
#
# Contract: the LLM writes a single self-contained `custom_op.asc` that
# registers with TORCH_LIBRARY / TORCH_LIBRARY_IMPL. This template — aligned
# with the modern CMake + torch.library sample in asc-devkit — compiles that
# file into libcustom_op.so. The evaluator then calls torch.ops.load_library
# on that file inside the PyTorch process.
#
# Deliberately NOT an OPP/custom_opp package and NOT a pybind extension:
#   - no npu_op_package / custom_opp_*.run
#   - no pybind11 / PYBIND11_MODULE
#   - CMAKE_SKIP_INSTALL_RULES is ON so `cmake --install` cannot write into
#     site-packages or $ASCEND_OPP_PATH/vendors
#   - RPATH is baked to $ORIGIN plus Torch / torch_npu / CANN lib dirs so
#     loading does not require mutating a shared global environment
#
# Requires: CANN >= 9.1.0 (source <cann>/set_env.sh), torch and torch_npu
# importable by the Python3 interpreter found by CMake.
#
# CMAKE_ASC_ARCHITECTURES is injected by build.py from the hardware profile
# (e.g. dav-2201, dav-3510).

cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED CMAKE_ASC_ARCHITECTURES OR "${CMAKE_ASC_ARCHITECTURES}" STREQUAL "")
    message(FATAL_ERROR "CMAKE_ASC_ARCHITECTURES is required (e.g. dav-2201, dav-3510)")
endif()
set(CMAKE_ASC_ARCHITECTURES "${CMAKE_ASC_ARCHITECTURES}" CACHE STRING "NPU architecture" FORCE)

# Never install this operator into the global Python or CANN tree.
set(CMAKE_SKIP_INSTALL_RULES ON)

find_package(ASC REQUIRED)
project(custom_op_build LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Optional ccache (same switch as the official ACLNN operator-project samples).
# Enable from the evaluator with -DENABLE_CCACHE=ON.
option(ENABLE_CCACHE "Enable ccache compiler launcher" OFF)
if(ENABLE_CCACHE)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        message(STATUS "[ccache] Enabled: ${CCACHE_PROGRAM}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    else()
        message(WARNING "[ccache] ENABLE_CCACHE=ON but ccache was not found")
    endif()
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

# Shared library loaded with torch.ops.load_library (not a pybind module).
add_library(custom_op SHARED custom_op.asc)

set_target_properties(custom_op PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "custom_op"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_include_directories(custom_op PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIRS}
)

target_link_libraries(custom_op PRIVATE
    ${TORCH_LIBRARIES}
    torch_npu
    Python3::Python
)

target_link_directories(custom_op PRIVATE
    ${TORCH_NPU_LIBRARIES}
)

# Runtime search path: the .so is loaded from the sample directory, so it
# must find Torch / torch_npu / CANN without a global install step.
set(_CUSTOM_OP_RPATH "$ORIGIN")
if(DEFINED TORCH_INSTALL_PREFIX AND NOT "${TORCH_INSTALL_PREFIX}" STREQUAL "")
    list(APPEND _CUSTOM_OP_RPATH "${TORCH_INSTALL_PREFIX}/lib")
endif()
if(EXISTS "${TORCH_NPU_PATH}/lib")
    list(APPEND _CUSTOM_OP_RPATH "${TORCH_NPU_PATH}/lib")
endif()
if(DEFINED ENV{ASCEND_HOME_PATH})
    if(EXISTS "$ENV{ASCEND_HOME_PATH}/lib64")
        list(APPEND _CUSTOM_OP_RPATH "$ENV{ASCEND_HOME_PATH}/lib64")
    endif()
    if(EXISTS "$ENV{ASCEND_HOME_PATH}/lib")
        list(APPEND _CUSTOM_OP_RPATH "$ENV{ASCEND_HOME_PATH}/lib")
    endif()
endif()
set_property(TARGET custom_op PROPERTY BUILD_RPATH ${_CUSTOM_OP_RPATH})
set_property(TARGET custom_op PROPERTY INSTALL_RPATH ${_CUSTOM_OP_RPATH})
set_property(TARGET custom_op PROPERTY BUILD_WITH_INSTALL_RPATH TRUE)
message(STATUS "custom_op RPATH: ${_CUSTOM_OP_RPATH}")

target_link_options(custom_op PRIVATE
    --gcc-toolchain=${GCC_TOOLCHAIN_ROOT}
)

target_compile_options(custom_op PRIVATE
    ${TORCH_CXX_FLAGS}
    $<$<COMPILE_LANGUAGE:ASC>:--gcc-toolchain=${GCC_TOOLCHAIN_ROOT}>
)
