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
    "${PROJECT_SOURCE_DIR}/process;${PROJECT_SOURCE_DIR}/utils;${PROJECT_SOURCE_DIR}/models"
    CACHE STRING "Include directories for ViennaPS GPU processing.")
set(VIENNAPS_CUDA_KERNELS
    "${ViennaRay_SOURCE_DIR}/gpu/kernels/normKernels.cu"
    CACHE STRING "Paths to the ViennaPS CUDA kernel source file.")
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
  include("${VIENNACORE_CMAKE_TOOLS}/cuda_helper.cmake")

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
