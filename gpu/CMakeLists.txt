#
# SPDX-FileCopyrightText: Copyright (c) 2019 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

project(ViennaPS-GPU)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

set_property(DIRECTORY PROPERTY EXCLUDE_FROM_ALL TRUE)

option(VIENNAPS_INPUT_ENABLE_OPTIXIR_SUPPORT
       "Enable support for generating OptiX-IR targeted input files" ON)

if(CUDA_VERSION VERSION_LESS 11.7)
  if(VIENNAPS_INPUT_ENABLE_OPTIXIR_SUPPORT)
    message(
      SEND_ERROR
        "VIENNAPS_INPUT_ENABLE_OPTIXIR_SUPPORT is not supported in CUDA versions less than 11.7")
  endif()
endif()

if(VIENNAPS_INPUT_ENABLE_OPTIXIR_SUPPORT)
  option(VIENNAPS_INPUT_GENERATE_OPTIXIR "Generate Optix-IR OptiX shaders" ON)
  option(VIENNAPS_INPUT_GENERATE_PTX "Generate PTX OptiX shaders" OFF)
else()
  option(VIENNAPS_INPUT_GENERATE_OPTIXIR "Generate Optix-IR OptiX shaders" OFF)
  option(VIENNAPS_INPUT_GENERATE_PTX "Generate PTX OptiX shaders" ON)
endif()

#### Set variables
set(VIENNAPS_GPU_INCLUDE_DIRS
    "${PROJECT_SOURCE_DIR}/process;${PROJECT_SOURCE_DIR}/rayTracing;${PROJECT_SOURCE_DIR}/utils;${PROJECT_SOURCE_DIR}/models"
    CACHE STRING "Include directories for ViennaPS GPU processing.")
set(VIENNAPS_CUDA_KERNELS
    "${PROJECT_SOURCE_DIR}/deviceKernels/normKernels.cu"
    CACHE STRING "Path to the ViennaPS CUDA kernel source file.")
set(VIENNAPS_PTX_DIR
    "${CMAKE_BINARY_DIR}/lib/ptx"
    CACHE STRING "Directory for compiled PTX files.")
set(CUDA_GENERATED_OUTPUT_DIR ${VIENNAPS_PTX_DIR})
set(CUDA_MIN_SM_TARGET
    sm_50
    CACHE STRING "Minimum CUDA SM architecture to use for compilation.")

if(WIN32)
  string(REPLACE "/" "\\\\" VIENNAPS_PTX_DIR ${VIENNAPS_PTX_DIR})
endif(WIN32)

# Present the CUDA_64_BIT_DEVICE_CODE on the default set of options.
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)

# Add some useful default arguments to the NVCC and NVRTC flags.  This is an example of
# how we use PASSED_FIRST_CONFIGURE.  Once you have configured, this variable is TRUE
# and following block of code will not be executed leaving you free to edit the values
# as much as you wish from the GUI or from ccmake.
if(NOT PASSED_FIRST_CONFIGURE)
  message(STATUS "Setting default NVCC flags for first time configuration.")
  include("../cmake/cuda_helper.cmake")

  set(CUDA_NVCC_FLAGS_DESCRIPTION "Semi-colon delimit multiple arguments.")
  string(REPLACE "sm_" "compute_" CUDA_MIN_SM_COMPUTE_TARGET ${CUDA_MIN_SM_TARGET})

  list(FIND CUDA_NVCC_FLAGS "-arch" index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS -arch ${CUDA_MIN_SM_TARGET})
    set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS}
        CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
  endif()

  add_cuda_flag("--use_fast_math")
  add_cuda_flag("-lineinfo")
  add_cuda_flag("--expt-relaxed-constexpr")
  add_cuda_flag("--generate-line-info")
  add_cuda_flag("-diag-suppress 20044")

  # CMAKE_CONFIGURATION_TYPES is only defined for multi-config build systems like
  # MSVC and Ninja, but we need to generate flags for each configuration
  # regardless.
  if(DEFINED CMAKE_CONFIGURATION_TYPES)
    set(OPTIX_CONFIGURATION_TYPES ${CMAKE_CONFIGURATION_TYPES})
  else()
    set(OPTIX_CONFIGURATION_TYPES "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
  endif()

  foreach(config ${OPTIX_CONFIGURATION_TYPES})
    if(${config} STREQUAL "Debug")
      add_cuda_flag_config(_${config} "-G")
      add_cuda_flag_config(_${config} "-O0")
    endif()
  endforeach()

  if(CUDA_VERSION VERSION_LESS "3.0")
    add_cuda_flag("--keep")
  endif()

  # Some CUDA 11.x toolkits erroneously complain about sm_50 being deprecated
  if(CUDA_VERSION VERSION_GREATER "11.0")
    add_cuda_flag("-Wno-deprecated-gpu-targets")
  endif()

  if(CUDA_USE_LOCAL_ENV)
    add_cuda_flag("--use-local-env")
  endif()
endif()

# In CMake, functions have their own scope, whereas macros use the scope of the caller.
macro(add_GPU_executable target_name_base target_name_var)
  set(target_name ${target_name_base})
  set(${target_name_var}
      ${target_name}
      PARENT_SCOPE)

  # Separate the sources from the CMake and CUDA options fed to the macro.  This code
  # comes from the CUDA_COMPILE_PTX macro found in FindCUDA.cmake.  We are copying the
  # code here, so that we can use our own name for the target.  target_name is used in the
  # creation of the output file names, and we want this to be unique for each target in
  # the SDK.
  cuda_get_sources_and_options(source_files cmake_options options ${ARGN})

  # Isolate OBJ target files. NVCC should only process these files and leave PTX targets for NVRTC
  set(cu_obj_source_files)
  set(cu_optix_source_files)
  foreach(file ${source_files})
    get_filename_component(_file_extension ${file} EXT)
    if(${_file_extension} MATCHES "cu")
      list(APPEND cu_optix_source_files ${file})
    endif()
  endforeach()

  # Add the path to the OptiX headers to our include paths.
  include_directories(${OptiX_INCLUDE})

  # Include ViennaPS headers which are used in pipelines
  include_directories(${VIENNAPS_GPU_INCLUDE_DIRS})
  include_directories(${ViennaCore_SOURCE_DIR}/include/viennacore) # needed for Context
  include_directories(${CMAKE_SOURCE_DIR}/include/viennaps/models)
  add_compile_definitions(VIENNACORE_COMPILE_GPU)

  # Create CUDA kernels
  cuda_wrap_srcs(
    ${target_name}
    PTX
    generated_files
    ${VIENNAPS_CUDA_KERNELS}
    ${cmake_options}
    OPTIONS
    ${options})

  # Create the rules to build the PTX and/or OPTIX files.
  if(VIENNAPS_INPUT_GENERATE_OPTIXIR)
    cuda_wrap_srcs(
      ${target_name}
      OPTIXIR
      generated_optixir_files
      ${cu_optix_source_files}
      ${cmake_options}
      OPTIONS
      ${options})
    list(APPEND generated_files ${generated_optixir_files})
  endif()
  if(VIENNAPS_INPUT_GENERATE_PTX)
    cuda_wrap_srcs(
      ${target_name}
      PTX
      generated_ptx_files
      ${cu_optix_source_files}
      ${cmake_options}
      OPTIONS
      ${options})
    list(APPEND generated_files ${generated_ptx_files})
  endif()

  # Here is where we create the rule to make the executable.  We define a target name and
  # list all the source files used to create the target.  In addition we also pass along
  # the cmake_options parsed out of the arguments.
  message(STATUS "Adding target: ${target_name}")
  add_executable(${target_name} ${source_files} ${generated_files} ${cmake_options})
  target_include_directories(${target_name} PRIVATE ${VIENNAPS_GPU_INCLUDE_DIRS})
  target_link_libraries(${target_name} ViennaPS ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})
  target_compile_definitions(${target_name}
                             PRIVATE VIENNAPS_KERNELS_PATH_DEFINE=${VIENNAPS_PTX_DIR})
endmacro()

if(VIENNAPS_BUILD_EXAMPLES)
  message(STATUS "[ViennaPS] Adding GPU Examples")
  add_subdirectory(examples)
  add_subdirectory(benchmark)
endif(VIENNAPS_BUILD_EXAMPLES)

if(VIENNAPS_BUILD_TESTS)
  message(STATUS "[ViennaPS] Adding GPU Tests")
  add_subdirectory(tests)
endif()

# Now that everything is done, indicate that we have finished configuring at least once.
# We use this variable to set certain defaults only on the first pass, so that we don't
# continually set them over and over again.
set(PASSED_FIRST_CONFIGURE
    ON
    CACHE INTERNAL "Already Configured once?")
